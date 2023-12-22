from Models.CoherenceNet import CoherenceNet
from Models.LitSegmentationModel import LitSegmentationModel

import numpy as np
from ipdb import set_trace
import sys, einops, torch, os
from argparse import ArgumentParser
from ShapeChecker import ShapeCheck
from functools import partial

class MethodeB(CoherenceNet) :
    """
    Model using as criterion the coherence of the optical flow in segmented regions :
    how an affine flow can explain the flow within a region. Using OLS to compute params
    """

    def __init__(self, theta_method,**kwargs) :
        super().__init__(**kwargs) # Build Coherence Net
        self.theta_method = theta_method
        self.hparams.update({'theta_method':theta_method})

    def ComputeTheta(self, pred, flow):
        """
        General Method to compute theta

        Params :
            pred (b, L, T(opt), I, J) : Mask proba predictions with optional time dimension
            flow ( b, 2, T(opt), I, J) : Flow Map with optional time dimension
        Returns :
            Theta : parameters set for each layers and sample : (b, L, ft)
        """
        if self.theta_method == 'OLS' :
            return self.ps.computetheta_ols(self.grid, flow, pred)
        if self.theta_method == 'Optim' :
            fctn = partial(self.CoherenceLoss, name=self.coherence_loss, vdist=self.vdist)
            return self.ps.computetheta_optim(self.grid, flow, pred, fctn)
        else :
            raise Exception(f'Method {self.theta_method} is not defined for theta computation')

    def ComputeParametricFlow(self, batch) :
        """
        For a given batch compute the parametric flow with the appropriate technique.
        Params :
            Batch containing at least :
                pred (b, L, T(opt), I, J) : Mask proba predictions
                flow ( b, 2, T(opt), I, J) : Original Flow Map ( before data augmentation )
        Returns :
            Add to batch 'Theta' ( b, l, ft) to the batch with the parametric motion parameters.
            param_flos (b, l, c, T(opt), i, j) : parametric flow for each layer
        """
        # theta_grad : Disable
        predv = batch['PredV'].detach()

        batch['Theta'] = self.ComputeTheta(predv, batch['FlowV'])

        batch['Theta'] = batch['Theta'].detach()

        param_flos = self.ps.parametric_flows(self.grid, batch['Theta'])
        return param_flos

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--theta_method', help='Method used to compute theta', type=str,
                            choices=['OLS','Optim'], default='Optim')
        return parser
