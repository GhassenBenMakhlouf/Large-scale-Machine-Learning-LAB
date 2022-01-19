import logging
import torch
from util.solver import Solver
from torch.optim.lr_scheduler import ExponentialLR
from pytorch_metric_learning import losses
from tqdm import tqdm


class SoundCLRSolver(Solver):
    """
    A specific Solver for training SoundCLR models.
    has some __init__ function as the Solver class with an extra parameter alpha.
    reimplements _reset(), train(), get_dataset_accuracy() and get_classwise_accuracy() functions.

    """

    def __init__(
        self,
        alpha=0.5,
        loss_type="HL",
        **kwargs,
    ):
        super(SoundCLRSolver, self).__init__(**kwargs)
        self.alpha = alpha
        self.loss_type = loss_type
        if self.loss_type == "HL":
            self.loss_func_2 = losses.SupConLoss(temperature=0.05)

    def _reset(self):
        """
        Set up some book-keeping variables for optimization.
        """
        self.best_model_stats = None
        self.best_params = None
        self.best_params_classifier = None
        self.best_params_projection = None

        self.train_loss_history = []
        self.val_loss_history = []

        self.current_patience = 0

    def train(self, epochs=100, patience=None):
        """
        Run optimization to train the model.
        """
        # Set up logging info
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        logging.info("starting training")
        # Start an epoch
        starting_epoch = 1
        if self.exp_resumed:
            starting_epoch = self.resume_epoch

        # learning rate decay rate like in the paper
        my_lr_scheduler = ExponentialLR(optimizer=self.opt, gamma=self.lr_decay)
        # Start an epoch
        for epoch in range(starting_epoch, epochs + 1):
            self.model.train()
            train_epoch_loss = 0.0
            train_samples_count = 0
            train_corrects = 0

            for batch in self.train_dataloader:
                # Unpack data
                inputs = batch[0]
                labels = batch[1]
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.opt.zero_grad()
                outputs = self.model(inputs)

                if self.loss_type == "HL":
                    train_ce_loss = self.loss_func(
                        outputs[:, : self.num_classes], labels
                    )
                    loss_2 = train_ce_loss * (1 - self.alpha)

                    train_cl_loss = self.loss_func_2(
                        outputs[:, self.num_classes :], labels
                    )
                    loss_1 = train_cl_loss * self.alpha

                    train_loss = loss_1 + loss_2
                    train_epoch_loss += train_loss.item()
                    train_loss.backward()
                    outputs = outputs[:, : self.num_classes]
                else:
                    train_loss = self.loss_func(outputs, labels)
                    train_epoch_loss += train_loss.item()
                    train_loss.backward()

                self.opt.step()
                train_corrects += (torch.argmax(outputs, dim=1) == labels).sum().item()
                train_samples_count += inputs.shape[0]

            self.writer.add_scalar("Learning rate", self.get_lr(), epoch)
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

            # validation
            if (epoch % self.validate_every) == 0:
                self.model.eval()
                val_loss = 0.0
                total, correct = 0, 0
                clswise_correct = [0 for i in range(self.num_classes)]
                clswise_total = [0 for i in range(self.num_classes)]

                for batch in self.val_dataloader:
                    # Unpack data
                    inputs = batch[0]
                    labels = batch[1]
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

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
                self.val_loss_history.append(val_loss)
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
                    logging.info(f"Stopping early at epoch {epoch}!")
                    break

                # Record the losses for later inspection.
                self.val_loss_history.append([val_loss, epoch])

        # At the end of training swap the best params into the model, projection head, and classifier
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
