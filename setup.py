import logging
from sacred import Experiment
import seml
import numpy as np
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.sampler import WeightedRandomSampler
from pytorch_metric_learning import losses
from sklearn.model_selection import train_test_split, StratifiedKFold
import random

from data import UrbanSoundDataset, BmwDataset
from tests import *
from models import (
    Transfer_Cnn14,
    TALNetV3Classifier,
    SoundCLR,
)
from util import (
    worker_init_fn,
    Solver,
    SoundCLRSolver,
    plot_classwise_prob,
    plot_conf_matrix,
)

ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


class ExperimentWrapper:
    """
    A simple wrapper around a sacred experiment, making use of sacred's captured functions with prefixes.
    This allows a modular design of the configuration, where certain sub-dictionaries (e.g., "data") are parsed by
    specific method. This avoids having one large "main" function which takes all parameters as input.
    """

    def __init__(self, n_folds=None, init_all=True):
        # Set up logging info
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        self.n_folds = n_folds
        if init_all:
            self.init_all()

    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.init_dataset()
        self.init_model()
        self.init_optimizer()

    # With the prefix option we can "filter" the configuration for the sub-dictionary under "data".
    @ex.capture(prefix="data")
    def init_dataset(
        self,
        dataset: str,
        test_percent: float,
        val_percent: float,
        batch_size: int,
        sample_rate: float,
        num_workers: int,
        split: str = "train",
        spectr_form: bool = False,
        window_size: int = 1024,
        hop_size: int = 512,
        waveform_aug: bool = False,
        spec_aug: bool = False,
        random_crop_size: int = 192000,
        random_scale: float = 1.25,
        n_freq_masks: int = 2,
        width_freq_masks: int = 32,
        n_time_masks: int = 1,
        width_time_masks: int = 32,
        mimic_bmw: bool = None,
        max_class_samples: int = None,
        weight_sample: bool = False,
    ):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        if dataset == "urban":
            self.num_classes = 10
            self.full_trainset = UrbanSoundDataset(
                split=split,
                mimic_bmw=mimic_bmw,
                resample=sample_rate,
                spectr_form=spectr_form,
                max_class_samples=max_class_samples,
                window_size=window_size,
                hop_size=hop_size,
                waveform_aug=waveform_aug,
                spec_aug=spec_aug,
                random_crop_size=random_crop_size,
                random_scale=random_scale,
                n_freq_masks=n_freq_masks,
                width_freq_masks=width_freq_masks,
                n_time_masks=n_time_masks,
                width_time_masks=width_time_masks,
            )
            testset = UrbanSoundDataset(
                split="test",
                train_set=self.full_trainset,
                resample=sample_rate,
                spectr_form=spectr_form,
                max_class_samples=max_class_samples,
                window_size=window_size,
                hop_size=hop_size,
                waveform_aug=waveform_aug,
                spec_aug=spec_aug,
                random_crop_size=random_crop_size,
                random_scale=random_scale,
                n_freq_masks=n_freq_masks,
                width_freq_masks=width_freq_masks,
                n_time_masks=n_time_masks,
                width_time_masks=width_time_masks,
            )
        elif dataset == "bmw":
            self.num_classes = 6
            self.full_trainset = BmwDataset(
                test_percent=test_percent,
                resample=sample_rate,
                train=True,
                max_class_samples=max_class_samples,
                spectr_form=spectr_form,
                window_size=window_size,
                hop_size=hop_size,
                waveform_aug=waveform_aug,
                spec_aug=spec_aug,
                random_crop_size=random_crop_size,
                random_scale=random_scale,
                n_freq_masks=n_freq_masks,
                width_freq_masks=width_freq_masks,
                n_time_masks=n_time_masks,
                width_time_masks=width_time_masks,
            )
            testset = BmwDataset(
                test_percent=test_percent,
                resample=sample_rate,
                train=False,
                max_class_samples=max_class_samples,
                spectr_form=spectr_form,
                window_size=window_size,
                hop_size=hop_size,
                waveform_aug=waveform_aug,
                spec_aug=spec_aug,
                random_crop_size=random_crop_size,
                random_scale=random_scale,
                n_freq_masks=n_freq_masks,
                width_freq_masks=width_freq_masks,
                n_time_masks=n_time_masks,
                width_time_masks=width_time_masks,
            )
        if self.n_folds is None:
            idx_train, idx_val = train_test_split(
                np.arange(len(self.full_trainset.labels)),
                test_size=val_percent,
                stratify=self.full_trainset.labels,
                random_state=1,
            )
            if weight_sample:
                trainset_labels = self.full_trainset.labels
                class_sample_count = np.array(
                    [
                        (trainset_labels == label).sum()
                        for label in np.unique(trainset_labels)
                    ]
                )
                weights = torch.tensor(1 / class_sample_count)
                samples_weights = torch.tensor(
                    [weights[label] for label in trainset_labels]
                )
                samples_weights[idx_val] = 0.0
                trainsampler = WeightedRandomSampler(
                    samples_weights, num_samples=len(idx_train)
                )
            else:
                trainsampler = SubsetRandomSampler(idx_train)
            valsampler = SubsetRandomSampler(idx_val)
            self.trainloader = DataLoader(
                self.full_trainset,
                batch_size=batch_size,
                sampler=trainsampler,
                num_workers=num_workers,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
            )
            self.valloader = DataLoader(
                self.full_trainset,
                batch_size=batch_size,
                sampler=valsampler,
                num_workers=num_workers,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
            )
            self.testloader = DataLoader(
                testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            )
        else:
            self.testloader = DataLoader(
                testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            )

    @ex.capture(prefix="model")
    def init_model(
        self, model_type: str, model_params: dict, pretrain: bool, pretrain_paths: dict,
    ):
        # Here we can pass the "model_params" dict to the constructor directly, which can be very useful in
        # practice, since we don't have to do any model-specific processing of the config dictionary.
        self.model_type = model_type
        assert model_params["num_classes"] == self.num_classes
        if model_type == "panns":
            assert model_params["sample_rate"] == self.sample_rate
            self.model = Transfer_Cnn14(**model_params)
            if pretrain:
                logging.info(
                    "Load pretrained model from {}".format(
                        pretrain_paths["panns_checkpoint_path"]
                    )
                )
                self.model.load_from_pretrain(pretrain_paths["panns_checkpoint_path"])
        elif model_type == "talnet":
            assert model_params["sample_rate"] == self.sample_rate
            self.model = TALNetV3Classifier(**model_params)
            if pretrain:
                logging.info(
                    "Load pretrained model from {}".format(
                        pretrain_paths["talnet_checkpoint_path"]
                    )
                )
                self.model.load_from_pretrain(pretrain_paths["talnet_checkpoint_path"])
        elif model_type == "soundCLR":
            self.model = SoundCLR(pretrain=pretrain, **model_params)

        # declare device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device {self.device}")
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)

    @ex.capture(prefix="optimization")
    def init_optimizer(
        self, loss_type: str, optimizer_type: str, optimizer_params: dict
    ):
        self.loss_type = loss_type
        if loss_type == "NLL":
            self.loss = nn.NLLLoss()
        elif loss_type == "CE":
            self.loss = nn.CrossEntropyLoss()
        elif loss_type == "HL":
            self.loss = nn.CrossEntropyLoss()
        elif loss_type == "CL":
            self.loss = losses.SupConLoss(temperature=0.05)

        if optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), **optimizer_params
            )
        else:
            raise NotImplementedError

    @ex.capture(prefix="training")
    def train(
        self,
        name: str,
        num_epochs: int,
        min_epochs: int,
        patience: int,
        lr_decay: float = 1.0,
        print_every: int = 1,
        validate_every: int = 1,
        checkpoint_interval: int = 1000,
        alpha: float = 0.5,
        update_criterion: str = "accuracy",
    ):
        logging.info(f"begin exp {name}")

        # everything is set up
        if self.model_type == "soundCLR":
            solver = SoundCLRSolver(
                model=self.model,
                train_dataloader=self.trainloader,
                val_dataloader=self.valloader,
                device=self.device,
                loss_func=self.loss,
                loss_type=self.loss_type,
                optimizer=self.optimizer,
                config_name=name,
                name=name,
                print_every=print_every,
                validate_every=validate_every,
                checkpoint_interval=checkpoint_interval,
                min_epochs=min_epochs,
                lr_decay=lr_decay,
                alpha=alpha,
                update_criterion=update_criterion,
            )
        else:
            solver = Solver(
                model=self.model,
                train_dataloader=self.trainloader,
                val_dataloader=self.valloader,
                device=self.device,
                loss_func=self.loss,
                optimizer=self.optimizer,
                config_name=name,
                name=name,
                print_every=print_every,
                validate_every=validate_every,
                checkpoint_interval=checkpoint_interval,
                min_epochs=min_epochs,
                lr_decay=lr_decay,
                update_criterion=update_criterion,
            )
        solver.train(num_epochs, patience)
        [
            test_acc,
            test_acc_top_two,
            test_acc_top_three,
            test_loss,
            classwise_accuracy,
            classwise_mean_prob,
            test_conf_matrix,
        ] = solver.get_dataset_stats(self.testloader)
        logging.info(f"Test accuracy: {test_acc}")
        logging.info(f"Test loss: {test_loss}")
        logging.info(f"Test second accuracy: {test_acc_top_two}")
        logging.info(f"Test third accuracy: {test_acc_top_three}")
        logging.info(f"Classwise Accuracy: {classwise_accuracy}")
        logging.info(f"Classwise Probability: {classwise_mean_prob}")
        logging.info(f"Test Confusion Matrix: {test_conf_matrix}")

        train_conf_matrix = solver.get_confusion_matrix(loader=self.trainloader)
        logging.info(f"Train Confusion Matrix: {train_conf_matrix}")
        val_conf_matrix = solver.get_confusion_matrix(loader=self.valloader)
        logging.info(f"Val Confusion Matrix: {val_conf_matrix}")

        plot_classwise_prob(
            probs=classwise_mean_prob,
            classes=self.testloader.dataset.classes,
            exp_name=name,
        )

        plot_conf_matrix(
            conf_matrix=train_conf_matrix,
            classes=np.arange(self.num_classes) + 1,
            exp_name=name,
            set="train",
        )
        plot_conf_matrix(
            conf_matrix=val_conf_matrix,
            classes=np.arange(self.num_classes) + 1,
            exp_name=name,
            set="val",
        )
        plot_conf_matrix(
            conf_matrix=test_conf_matrix,
            classes=np.arange(self.num_classes) + 1,
            exp_name=name,
            set="test",
        )

        results = {
            "test_acc": test_acc,
            "test_loss": test_loss,
            "test_second_acc": test_acc_top_two,
            "test_third_acc": test_acc_top_three,
            "classwise_accuracy": classwise_accuracy,
            "mean_prob": classwise_mean_prob,
        }

        return results

    @ex.capture(prefix="training")
    def train_kfold(
        self,
        name: str,
        num_epochs: int,
        min_epochs: int,
        patience: int,
        lr_decay: float = 1.0,
        print_every: int = 1,
        validate_every: int = 1,
        checkpoint_interval: int = 1000,
        alpha: float = 0.5,
        update_criterion: str = "accuracy",
    ):
        full_trainset_size = len(self.full_trainset)
        indices = list(range(full_trainset_size))
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=1)
        test_acc_folds = []
        test_loss_folds = []
        test_second_acc_folds = []
        test_third_acc_folds = []
        classwise_accuracy_folds = []
        classwise_mean_prob_folds = []
        test_conf_matrix_folds = []
        fold = 1
        for train_indices, val_indices in skf.split(
            X=indices, y=self.full_trainset.labels
        ):
            trainsampler = SubsetRandomSampler(train_indices)
            valsampler = SubsetRandomSampler(val_indices)
            self.trainloader = DataLoader(
                self.full_trainset,
                batch_size=self.batch_size,
                sampler=trainsampler,
                num_workers=self.num_workers,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
            )
            self.valloader = DataLoader(
                self.full_trainset,
                batch_size=self.batch_size,
                sampler=valsampler,
                num_workers=self.num_workers,
                pin_memory=True,
                worker_init_fn=worker_init_fn,
            )

            if self.model_type == "soundCLR":
                solver = SoundCLRSolver(
                    model=self.model,
                    train_dataloader=self.trainloader,
                    val_dataloader=self.valloader,
                    device=self.device,
                    loss_func=self.loss,
                    loss_type=self.loss_type,
                    optimizer=self.optimizer,
                    config_name=name,
                    name=f"{name}_fold{fold}",
                    print_every=print_every,
                    validate_every=validate_every,
                    checkpoint_interval=checkpoint_interval,
                    min_epochs=min_epochs,
                    lr_decay=lr_decay,
                    alpha=alpha,
                    update_criterion=update_criterion,
                )
            else:
                solver = Solver(
                    model=self.model,
                    train_dataloader=self.trainloader,
                    val_dataloader=self.valloader,
                    device=self.device,
                    loss_func=self.loss,
                    loss_type=self.loss_type,
                    optimizer=self.optimizer,
                    config_name=name,
                    name=f"{name}_fold{fold}",
                    print_every=print_every,
                    validate_every=validate_every,
                    checkpoint_interval=checkpoint_interval,
                    min_epochs=min_epochs,
                    lr_decay=lr_decay,
                    update_criterion=update_criterion,
                )
            solver.train(num_epochs, patience)
            [
                test_acc,
                test_acc_top_two,
                test_acc_top_three,
                test_loss,
                classwise_accuracy,
                classwise_mean_prob,
                test_conf_matrix,
            ] = solver.get_dataset_stats(self.testloader)
            logging.info(f"Test accuracy_fold{fold}: {test_acc}")
            logging.info(f"Test loss_fold{fold}: {test_loss}")
            logging.info(f"Test second accuracy fold{fold}: {test_acc_top_two}")
            logging.info(f"Test third accuracy fold{fold}: {test_acc_top_three}")
            logging.info(f"Classwise Accuracy_fold{fold}: {classwise_accuracy}")
            logging.info(f"Classwise Probability_fold{fold}: {classwise_mean_prob}")
            logging.info(f"Test Confusion Matrix_fold{fold}: {test_conf_matrix}")

            train_conf_matrix = solver.get_confusion_matrix(loader=self.trainloader)
            logging.info(f"Train Confusion Matrix_fold{fold}: {train_conf_matrix}")
            val_conf_matrix = solver.get_confusion_matrix(loader=self.valloader)
            logging.info(f"Val Confusion Matrix_fold{fold}: {val_conf_matrix}")
            test_acc_folds.append(test_acc)
            test_loss_folds.append(test_loss)
            test_second_acc_folds.append(test_acc_top_two)
            test_third_acc_folds.append(test_acc_top_three)
            classwise_accuracy_folds.append(classwise_accuracy)
            classwise_mean_prob_folds.append(classwise_mean_prob)
            test_conf_matrix_folds.append(test_conf_matrix)

            fold += 1
            self.init_model()
            self.init_optimizer()

        avg_test_acc = sum(test_acc_folds) / len(test_acc_folds)
        avg_test_loss = sum(test_loss_folds) / len(test_loss_folds)
        avg_test_second_acc = sum(test_second_acc_folds) / len(test_second_acc_folds)
        avg_test_third_acc = sum(test_third_acc_folds) / len(test_third_acc_folds)
        avg_classwise_prob = sum(classwise_mean_prob_folds) / len(
            classwise_mean_prob_folds
        )
        avg_test_matrix = sum(test_conf_matrix_folds) / len(test_conf_matrix_folds)
        checkpoints_dir = Path("checkpoints") / name
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)
        plot_classwise_prob(
            probs=avg_classwise_prob,
            classes=self.testloader.dataset.classes,
            exp_name=f"{name}",
        )
        plot_conf_matrix(
            conf_matrix=avg_test_matrix,
            classes=np.arange(self.num_classes) + 1,
            exp_name=f"{name}",
            set=f"test",
        )

        avg_classwise_accuracy = [
            sum(i) / len(classwise_accuracy_folds)
            for i in zip(*classwise_accuracy_folds)
        ]
        avg_mean_prob = sum(classwise_mean_prob_folds) / len(classwise_mean_prob_folds)

        logging.info(f"Average Test Accuracy: {avg_test_acc}")
        logging.info(f"Average Test Loss: {avg_test_loss}")
        logging.info(f"Average Test second Accuracy: {avg_test_second_acc}")
        logging.info(f"Average Test third Accuracy: {avg_test_third_acc}")
        logging.info(f"Average Class wise Accuracy: {avg_classwise_accuracy}")
        logging.info(f"Average Mean Probabilities: {avg_mean_prob}")

        results = {
            "avg_test_acc": avg_test_acc,
            "avg_test_loss": avg_test_loss,
            "avg_test_second_acc": avg_test_second_acc,
            "avg_test_third_acc": avg_test_third_acc,
            "avg_classwise_accuracy": avg_classwise_accuracy,
            "avg_mean_prob": avg_mean_prob,
            "test_acc_folds": test_acc_folds,
            "test_loss_folds": test_loss_folds,
            "classwise_accuracy_folds": classwise_accuracy_folds,
            "classwise_mean_prob_folds": classwise_mean_prob_folds,
        }

        return results


# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@ex.command(unobserved=True)
def get_experiment(init_all=False):
    print("get_experiment")
    experiment = ExperimentWrapper(init_all=init_all)
    return experiment


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@ex.automain
@ex.capture(prefix="cross_validation")
def train(n_folds: int = None, experiment=None):
    if experiment is None:
        if n_folds is None:
            experiment = ExperimentWrapper()
            return experiment.train()
        else:
            experiment = ExperimentWrapper(n_folds=n_folds)
            return experiment.train_kfold()
