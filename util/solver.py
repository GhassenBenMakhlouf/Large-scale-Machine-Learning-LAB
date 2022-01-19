import os
import logging
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy
from shutil import copyfile
from datetime import datetime


class Solver(object):
    """
    A general Solver for training models

    The solver accepts both training and validataion data and labels so it can
    periodically check classification accuracy on both training and validation
    data.

    After the train() method returns, the model will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variables solver.train_loss_history and
    solver.val_loss_history will be lists containing the losses of the model
    on the training and validation set at each epoch.
    """

    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        device,
        loss_func,
        optimizer,
        config_name,
        name,
        verbose=True,
        print_every=1,
        validate_every=1,
        min_epochs=100,
        resume_epoch=None,  # e.g. "100"
        resume_exp=None,  # e.g. "exp6"
        lr_decay=1.0,
        checkpoint_interval=1000,
        update_criterion="accuracy",  # "accuracy" or "loss"
        **kwargs,
    ):
        """
        Construct a new Solver instance.

        Required arguments:
        - model: A model object conforming to the API described above
        - cfg: An EasyDict created from config.yaml
        - train_dataloader: A generator object returning training data
        - val_dataloader: A generator object returning validation data
        - device: The device currently working on
        - loss_func: Loss function object.
        - optimizer: The optimizer specifying the update rule

        Optional arguments:
        - verbose: Boolean; if set to false then no output will be printed during
          training.
        - print_every: Integer; training losses will be printed every print_every
          iterations.
        """
        self.model = model
        self.loss_func = loss_func
        self.opt = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device

        self.config_name = config_name
        self.name = name
        self.verbose = verbose
        self.print_every = print_every
        self.validate_every = validate_every
        self.min_epochs = min_epochs
        self.lr_decay = lr_decay
        self.checkpoint_interval = checkpoint_interval
        self.update_criterion = update_criterion

        self.num_classes = int(len(self.train_dataloader.dataset.classes))

        log_dir = Path("logs")
        checkpoints_dir = Path("checkpoints")

        if bool(resume_epoch) != bool(resume_exp):
            logging.error(f"Please specify both resume_epoch and resume_exp!")
            exit()
        self.exp_resumed = False
        if resume_exp:
            assert self.name == resume_exp
            self.this_exp = str(resume_exp)
            this_exp_path = log_dir / self.this_exp
            self.checkpoints_dir = checkpoints_dir / self.this_exp
            checkpoints = [x.name for x in sorted(self.checkpoints_dir.glob("*.pth"))]
            model_path = (
                self.checkpoints_dir / f"{self.this_exp}_epoch{resume_epoch}.pth"
            )
            if f"{self.this_exp}_epoch{resume_epoch}.pth" in checkpoints:
                self.model.load_state_dict(torch.load(model_path))
                logging.info(f"checkpoint {model_path} is loaded.")
                self.exp_resumed = True
                self.resume_epoch = resume_epoch
            else:
                logging.error(f"checkpoint {model_path} can not be found!")
                exit()

        if not self.exp_resumed:
            exp_n = len(list(x.name for x in sorted(log_dir.glob("exp*"))))
            self.this_exp = str("exp" + str(exp_n + 1))
            this_exp_path = log_dir / self.name
            self.checkpoints_dir = Path("checkpoints") / self.name
            if not os.path.exists(self.checkpoints_dir):
                os.mkdir(self.checkpoints_dir)

        self.writer = SummaryWriter(log_dir=this_exp_path)
        copyfile(
            f"configs/{self.config_name}.yaml",
            f"{self.checkpoints_dir}/{self.config_name}.yaml",
        )

        self._reset()

    def _reset(self):
        """
        Set up some book-keeping variables for optimization.
        """
        self.best_model_stats = None
        self.best_params = None

        self.train_loss_history = []
        self.val_loss_history = []

        self.current_patience = 0

    def train(self, epochs=100, patience=None):
        """
        Run optimization to train the model.
        """
        # Set up logging info
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        # Start an epoch
        starting_epoch = 1
        if self.exp_resumed:
            starting_epoch = self.resume_epoch
        # learning rate decay with rate gamma
        my_lr_scheduler = ExponentialLR(optimizer=self.opt, gamma=self.lr_decay)

        for epoch in range(starting_epoch, epochs + 1):
            # Iterate over all training samples
            train_epoch_loss = 0.0
            self.model.train()

            train_samples_count = 0
            train_corrects = 0

            for batch in self.train_dataloader:
                # Unpack data
                inputs = batch[0]
                labels = batch[1]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Update the model parameters.
                self.opt.zero_grad()
                outputs = self.model(inputs)
                train_loss = self.loss_func(outputs, labels)
                train_epoch_loss += train_loss.item()
                train_loss.backward()
                self.opt.step()

                train_corrects += (torch.argmax(outputs, dim=1) == labels).sum().item()
                train_samples_count += inputs.shape[0]

            self.writer.add_scalar("Learning rate", self.get_lr(), epoch)
            # lr decay
            my_lr_scheduler.step()

            train_epoch_loss /= len(self.train_dataloader)
            train_accuracy = 100 * train_corrects / train_samples_count
            self.train_loss_history.append(train_epoch_loss)
            self.writer.add_scalar("Loss/Train", train_epoch_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
            if self.verbose and (epoch % self.print_every) == 0:
                logging.info(
                    f"[{epoch:03d}] train_loss: {train_epoch_loss:.3f}, train_accuracy: {train_accuracy:.3f}%"
                )
            self.train_loss_history.append(train_epoch_loss)

            if epoch % self.checkpoint_interval == 0:
                torch.save(
                    self.model.state_dict(),
                    f"{self.checkpoints_dir}/{self.name}_model_epoch{epoch}.pth",
                )
                logging.info(f"Saved model checkpoint {epoch}")

            if (epoch % self.validate_every) == 0:
                # Iterate over all validation samples
                self.model.eval()
                val_loss = 0.0
                total, correct = 0, 0
                clswise_correct = [0 for i in range(self.num_classes)]
                clswise_total = [0 for i in range(self.num_classes)]
                for batch_val in self.val_dataloader:
                    # Unpack data
                    inputs = batch_val[0]
                    labels = batch_val[1]
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # Compute Loss - no param update at validation time!
                    with torch.no_grad():
                        outputs = self.model(inputs)

                    predicted_labels = torch.argmax(outputs, dim=1)

                    for i in range(self.num_classes):
                        clswise_correct[i] += (
                            sum((predicted_labels == labels) * (labels == i))
                            .sum()
                            .item()
                        )
                        clswise_total[i] += (labels == i).sum().item()

                    total += predicted_labels.numel()
                    correct += (predicted_labels == labels).sum().item()
                    val_loss += self.loss_func(outputs, labels).item()

                val_loss = val_loss / len(self.val_dataloader)
                val_accuracy = 100 * correct / total
                self.writer.add_scalar("Loss/Val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/Val", val_accuracy, epoch)
                clswise_accuracy = [0 for i in range(self.num_classes)]
                for i in range(self.num_classes):
                    if clswise_total[i] != 0:
                        clswise_accuracy[i] = clswise_correct[i] / clswise_total[i]
                        self.writer.add_scalar(
                            f"Accuracy/Val/{self.val_dataloader.dataset.classes[i]}",
                            clswise_accuracy[i],
                            epoch,
                        )

                if self.verbose:
                    logging.info(
                        f"[{epoch:03d}] val_loss: {val_loss :.3f}, val_accuracy: {val_accuracy:.3f}%"
                    )
                    logging.info(
                        f"[{epoch:03d}] val_clswise_accuracy: {clswise_accuracy}"
                    )

                # Keep track of the best model
                if self.update_criterion == "accuracy":
                    self.update_best_accuracy(val_accuracy, val_loss)
                elif self.update_criterion == "loss":
                    self.update_best_loss(val_accuracy, val_loss)
                else:
                    raise NotImplementedError(
                        f'update_criterion can only be "accuracy" or "loss". "{self.update_criterion}" is not implemented.'
                    )
                if (
                    epoch > self.min_epochs
                    and patience
                    and self.current_patience >= patience
                ):
                    logging.info("Stopping early at epoch {}!".format(epoch))
                    break

                # Record the losses for later inspection.
                self.val_loss_history.append([val_loss, epoch])

        # At the end of training swap the best params into the model
        self.model.load_state_dict(self.best_params)
        torch.save(
            self.model.state_dict(),
            f"{self.checkpoints_dir}/{self.name}_final_model.pth",
        )
        logging.info(
            f"Saved final {self.name} model with validation accuracy {self.best_model_stats['accuracy']}"
        )
        # to make sure that all pending events have been written to disk
        self.writer.flush()

    def update_best_accuracy(self, accuracy, val_loss):
        # Update the model and best accuracy if we see improvements.
        if not self.best_model_stats or accuracy > self.best_model_stats["accuracy"]:
            self.best_model_stats = {"accuracy": accuracy, "val_loss": val_loss}
            self.best_params = deepcopy(self.model.state_dict())
            self.current_patience = 0
        else:
            self.current_patience += 1

    def update_best_loss(self, accuracy, val_loss):
        # Update the model and best loss if we see improvements.
        if not self.best_model_stats or val_loss < self.best_model_stats["val_loss"]:
            self.best_model_stats = {"accuracy": accuracy, "val_loss": val_loss}
            self.best_params = deepcopy(self.model.state_dict())
            self.current_patience = 0
        else:
            self.current_patience += 1

    def get_dataset_accuracy(self, loader):
        self.model.eval()
        correct = 0
        total = 0
        loss = 0
        for batch in loader:
            inputs = batch[0]
            labels = batch[1]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)

            pred = torch.argmax(outputs, dim=1)
            loss += self.loss_func(outputs, labels).item()

            correct += sum(pred == labels)
            total += labels.numel()

        accuracy = 100 * correct / total
        loss = loss / len(loader)
        return [accuracy, loss]

    def get_classwise_accuracy(self, loader):
        self.model.eval()
        correct = [0 for i in range(self.num_classes)]
        total = [0 for i in range(self.num_classes)]
        accuracy = [0 for i in range(self.num_classes)]
        for batch in loader:
            inputs = batch[0]
            labels = batch[1]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
            pred = torch.argmax(outputs, dim=1)
            for i in range(self.num_classes):
                correct[i] += sum((pred == labels) * (labels == i)).sum().item()
                total[i] += (labels == i).sum().item()

        for i in range(self.num_classes):
            if total[i] != 0:
                accuracy[i] = correct[i] / total[i]
        return accuracy

    def get_lr(self):
        for param_group in self.opt.param_groups:
            return param_group["lr"]

    def get_classwise_prob(self, loader):
        self.model.eval()
        total = torch.zeros(self.num_classes, device=self.device)
        acc_prob = torch.zeros(
            self.num_classes, self.num_classes, device=self.device
        )  # each row contains the accumulated probabilities of each class
        for batch in loader:
            inputs = batch[0]
            labels = batch[1]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)

            prob = torch.softmax(outputs, dim=1)

            # prob = torch.sum(prob, dim=0)
            for i in range(prob.shape[0]):
                acc_prob[labels[i], :] += prob[i, :]
                total[labels[i]] += 1

        mean_prob = torch.zeros(self.num_classes, self.num_classes, device=self.device)

        for i in range(self.num_classes):
            if total[i] != 0:
                mean_prob[i, :] = acc_prob[i, :] / total[i]

        return mean_prob * 100

    def get_confusion_matrix(self, loader):
        self.model.eval()
        predictions = []
        true_labels = []
        for batch in loader:
            inputs = batch[0]
            labels = batch[1]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                outputs = self.model(inputs)
            pred = torch.argmax(outputs, dim=1)

            predictions += pred.tolist()
            true_labels += labels.tolist()

        conf_matrix = confusion_matrix(
            y_true=true_labels, y_pred=predictions, labels=list(range(6)),
        )
        return conf_matrix

    def get_dataset_n_accuracy(self, loader, n):
        assert n < self.num_classes
        self.model.eval()
        correct = 0
        total = 0
        for batch in loader:
            inputs = batch[0]
            labels = batch[1]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)

            _, indices = torch.sort(outputs, descending=True, dim=1)
            top_classes = indices[:, :n]
            correct += sum(
                [label in top_classes[i, :] for i, label in enumerate(labels)]
            )
            total += labels.numel()

        n_accuracy = 100 * correct / total
        return n_accuracy

    def get_dataset_stats(self, loader):
        self.model.eval()
        correct, correct_top_two, correct_top_three = 0, 0, 0
        total = 0
        loss = 0
        cls_correct = [0 for i in range(self.num_classes)]
        cls_total = [0 for i in range(self.num_classes)]
        cls_accuracy = [0 for i in range(self.num_classes)]
        cls_prob = torch.zeros(
            self.num_classes, self.num_classes, device=self.device
        )  # each row contains the accumulated probabilities of each class
        mean_prob = torch.zeros(self.num_classes, self.num_classes, device=self.device)
        predictions = []
        true_labels = []
        for batch in loader:
            inputs = batch[0]
            labels = batch[1]
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)

            pred = torch.argmax(outputs, dim=1)
            prob = torch.softmax(outputs, dim=1)
            _, pred_top_two = torch.topk(outputs, k=2, dim=1)
            _, pred_top_three = torch.topk(outputs, k=3, dim=1)
            loss += self.loss_func(outputs, labels).item()

            for i in range(self.num_classes):
                cls_correct[i] += sum((pred == labels) * (labels == i)).sum().item()
                cls_total[i] += (labels == i).sum().item()
            for i in range(prob.shape[0]):
                cls_prob[labels[i], :] += prob[i, :]

            correct += (pred == labels).sum().item()
            correct_top_two += (pred_top_two == labels.unsqueeze(dim=1)).sum().item()
            correct_top_three += (
                (pred_top_three == labels.unsqueeze(dim=1)).sum().item()
            )
            total += labels.numel()
            predictions += pred.tolist()
            true_labels += labels.tolist()

        accuracy = 100 * correct / total
        accuracy_top_two = 100 * correct_top_two / total
        accuracy_top_three = 100 * correct_top_three / total
        loss = loss / len(loader)
        for i in range(self.num_classes):
            if cls_total[i] != 0:
                cls_accuracy[i] = cls_correct[i] / cls_total[i]
                mean_prob[i, :] = cls_prob[i, :] / cls_total[i]
        conf_matrix = confusion_matrix(
            y_true=true_labels,
            y_pred=predictions,
            labels=list(range(self.num_classes)),
        )
        return [
            accuracy,
            accuracy_top_two,
            accuracy_top_three,
            loss,
            cls_accuracy,
            mean_prob * 100,
            conf_matrix,
        ]
