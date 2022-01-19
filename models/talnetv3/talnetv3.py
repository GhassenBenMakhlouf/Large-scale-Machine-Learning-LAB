import os
import logging
import torch
import torch.nn as nn
from argparse import ArgumentParser
from models.talnetv3._talnet import TALNetV3NoMeta


class TALNetV3Classifier(nn.Module):
    def __init__(
        self,
        num_classes,
        sample_rate,
        num_mels,
        window_size,
        hop_size,
        fmin,
        fmax,
        spec_aug,
        dropout=0.0,
        dropout_transfo=0.2,
        dropout_AS=0.0,
        n_conv_layers=10,
        n_pool_layers=5,
        kernel_size="3",
        pooling="att",
        embedding_size=1024,
        batch_norm=True,
        transfo_head=16,
    ):
        super().__init__()

        # Save hparams for later

        self.num_classes = num_classes
        self.input_size = (162, num_mels)
        self.best_scores = [0] * 5

        self.model = TALNetV3NoMeta(
            num_classes,
            sample_rate,
            num_mels,
            window_size,
            hop_size,
            fmin,
            fmax,
            spec_aug,
            dropout,
            dropout_transfo,
            dropout_AS,
            n_conv_layers,
            n_pool_layers,
            kernel_size,
            pooling,
            embedding_size,
            batch_norm,
            transfo_head,
        )

    def load_from_pretrain(self, pretrained_checkpoint_path):
        # Load every pretrained layers matching our layers
        pretrained_dict = torch.load(pretrained_checkpoint_path, map_location="cpu")[
            "model"
        ]
        model_dict = self.model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.size() == model_dict[k].size()
        }
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.model(x)[0]
        return x
