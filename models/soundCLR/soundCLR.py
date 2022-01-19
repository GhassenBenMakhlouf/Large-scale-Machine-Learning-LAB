import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import logging


class SoundCLR(nn.Module):
    """
    SoundCLR variable Module according to:
    https://arxiv.org/pdf/2103.01929v1.pdf
    """

    def __init__(
        self,
        num_classes,
        loss_type="HL",
        pretrain=True,
    ):
        super().__init__()
        assert loss_type in ["HL", "CE", "CL"]
        self.loss_type = loss_type
        self.base = torchvision.models.resnet50(pretrained=pretrain)
        if pretrain:
            logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
            logging.info("Load pretrained resnet50 model from torchvision models.")
        self.base.fc = nn.Identity()
        if self.loss_type == "HL":
            self.cls_layer = nn.Linear(2048, num_classes)
            self.proj_layer = nn.Linear(2048, 64)
        elif self.loss_type == "CE":
            self.cls_layer = nn.Linear(2048, num_classes)
        elif self.loss_type == "CL":
            self.proj_layer = nn.Linear(2048, 64)

    def forward(self, x):

        x = self.base(x)
        if self.training:
            x = F.normalize(x, dim=0)
        if self.loss_type == "HL":
            classifcation = self.cls_layer(x)
            if self.training:
                projection = self.proj_layer(x)
                projection = F.normalize(projection, dim=0)
                return torch.cat((classifcation, projection), dim=1)
            else:
                return classifcation
        elif self.loss_type == "CE":
            classification = self.cls_layer(x)
            return classification
        elif self.loss_type == "CL":
            projection = self.proj_layer(x)
            projection = F.normalize(projection, dim=0)
            return projection
