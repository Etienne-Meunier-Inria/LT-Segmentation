import pytorch_lightning as pl
from ipdb import set_trace
from Models.Backbones.FlowUnet3D import FlowUnet3D
from Models.Backbones.generalunet.u3D_3D import Unet3D_3D
from Models.Backbones.generalunet.u3D_3D_MaskFormer import MaskFormer3D

from Models.Backbones.generalunet.unet_clean import UNetClean
from argparse import ArgumentParser
import torch
from torchvision import models
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from ipdb import set_trace


class LitBackbone(pl.LightningModule):

    def __init__(self, inputs, backbone, num_classes, hparams, **kwargs) :
        super().__init__()
        self.model = self.init_model(backbone = backbone,
                                     inputs = inputs,
                                     num_classes = num_classes, hparams=hparams)

    def init_model(self, *, backbone, num_classes, inputs, hparams) :
        print(f'Initialise backbone {backbone}')
        if backbone == 'FlowUnet3D' :
            return FlowUnet3D(Unet3D_3D, inputs, num_classes, hparams)
        elif backbone == 'MaskFormer3D' :
            return FlowUnet3D(MaskFormer3D, inputs, num_classes, hparams)
        else :
            print(f'Backbone {backbone} not available')

    def forward(self, batch) :
        return self.model(batch)

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = UNetClean.add_specific_args(parser)
        parser = MaskFormer3D.add_specific_args(parser)
        parser.add_argument('--backbone', '-bb', type=str, choices=['FlowUnet3D', 'MaskFormer3D'], default='MaskFormer3D')
        parser.add_argument('--inputs', nargs='+', type=str, default=['Flow-1', 'Flow', 'Flow+1'])
        return parser
