from generalunet.unet_clean import UNetClean
from generalunet.u3D_3D import Unet3D_3D
from generalunet.utils.ArgsUtils import argf
import torch.nn as nn
import einops
import torch
from utils.ExperimentalFlag import ExperimentalFlag as Ef
from utils.ExperimentalFlag import *


from ipdb import set_trace
from torchvision.transforms import RandomErasing


class FlowUnet3D(nn.Module) :
    def __init__(self, base_net, inputs, num_classes, hparams) :
        super().__init__()
        self.inputs = inputs
        self.model = base_net(input_channels=2, num_classes=num_classes, **argf(hparams, 'unet'), **argf(hparams, 'maskformer'))

    def forward(self, batch) :
        """
        Args :
            batch dict 'Flow' (b c h w): dictionnary with input keys ( Flow at different timesteps )
        Returns :
            predv ( b l t h w) : Preds on which the loss is applied
            target_index : index of the target frame in the temporal dimension
            Add to batch :
                HiddenV (unet.num_layers b ftd td hd wd) : List hidden representation at the bottleneck

        """
        input = einops.rearrange([batch[k] for k in self.inputs], 't b c h w -> b c t h w')
        if Ef.check('PerturbInputFlowNoise') :
            input = perturb_flow_noise(input, p=0.2, max_step=3)
        if Ef.check('PerturbInputFlowNull') :
            input = perturb_flow_null(input, p=0.2, max_step=3)
        if Ef.check('PerturbInputFlowErase') :
            input = perturb_flow_random_erase(input, p=0.2, max_step=3)
        pred, auxs = self.model(input)
        batch.update(auxs)
        pred = torch.softmax(pred, axis=1)
        batch['InputV'] = input.detach()
        return pred, self.inputs.index('Flow')

def perturb_flow_noise(flowv, p=0.2, max_step=3) :
    if torch.rand(1)[0] > (1-p):
        idx = get_index(flowv.shape[2], max_step)
        print(f'Perturb flow noise : {idx}')
        flowv[:,:,idx] = torch.randn_like(flowv[:,:,idx]) * flowv[:,:,idx].var(axis=(-2,-1), keepdims=True) +\
                         flowv[:,:,idx].mean(axis=(-2,-1), keepdims=True)
    return flowv

def perturb_flow_null(flowv, p=0.2, max_step=3) :
    if torch.rand(1)[0] > (1-p):
        idx = get_index(flowv.shape[2], max_step)
        print(f'Perturb flow null : {idx}')
        flowv[:,:,idx] = torch.randn_like(flowv[:,:,idx])
    return flowv

def perturb_flow_random_erase(flowv, p=0.2, max_step=3) :
    if torch.rand(1)[0] > (1-p):
        idx = get_index(flowv.shape[2], max_step)
        print(f'Perturb flow erase : {idx}')
        re = RandomErasing(p=1,
                           scale=(0.02, 0.3),
                           ratio=(0.3, 3.3),
                           value='random')
        flowv[:,:,idx] = re(flowv[:,:,idx])
    return flowv

def get_index(max_index, max_step) :
    s = torch.randint(min(max_index, max_step), (1,)).item()
    s = max(s,1)
    idx_start = torch.randint(max_index-s, (1,)).item()
    idx = [idx_start] + [idx_start+si for si in range(s)[1:]]
    return idx
