import flowiz, torch, torch.nn as nn, torch.optim as optim, pytorch_lightning as pl, numpy as np
from PIL import Image
from argparse import Namespace, ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
from evaluations import *
from Models.LitBackbone import LitBackbone
from functools import partial

class LitSegmentationModel(pl.LightningModule) :
    def __init__(self, nAffineMasks, **kwargs) :
        super().__init__()
        self.hparams.update(kwargs)
        self.hparams.update({'nAffineMasks':nAffineMasks})
        self.backbone_model = LitBackbone(num_classes=nAffineMasks, hparams=self.hparams, **kwargs)
        self.d = Namespace(**{'L': nAffineMasks})
        self.request = set(self.backbone_model.model.inputs)

    def configure_optimizers(self):
        d = {}
        if self.hparams['optim.name'] == 'Adam':
            d['optimizer'] = optim.Adam(self.parameters(), lr=self.hparams['optim.lr'],
             weight_decay=self.hparams['optim.weight_decay'])
        else :
             raise Exception(f"Optimizer {self.hparams['optim.name']} not available")

        print(f'Optimizer : {d["optimizer"]}')

        if self.hparams['optim.schedule'] == 'StepLR' :
            d['lr_scheduler'] = { "scheduler": optim.lr_scheduler.StepLR(d['optimizer'], step_size=50, gamma=0.1),
                               "interval": "epoch",
                               "frequency": 1}
        elif self.hparams['optim.schedule'] == 'CyclicLR' :
            d['lr_scheduler'] = { "scheduler": optim.lr_scheduler.CyclicLR(d['optimizer'],
                                                     base_lr=self.hparams['optim.lr']/100,
                                                     max_lr=self.hparams['optim.lr'],
                                                     cycle_momentum=False,
                                                     step_size_up=50, step_size_down=50),
                               "interval": "epoch",
                               "frequency": 1}
        elif self.hparams['optim.schedule'] == 'WarmupDecay' :
            def warmupdecay(iteration, end_warmup, decay_step) :
                if iteration <= end_warmup :
                    return iteration/end_warmup
                else :
                    return (0.5)**((iteration - end_warmup)//decay_step)
            scheduler = partial(warmupdecay, end_warmup=20, decay_step=40)
            d['lr_scheduler'] = { "scheduler": optim.lr_scheduler.LambdaLR(d['optimizer'], lr_lambda=scheduler),
                               "interval": "epoch",
                               "frequency": 1}
        return d

    def prediction(self, batch) :
        """
        Produce a prediction for a segmentation using the model and the batch
        """
        batch['PredV'], index_target = self.backbone_model.forward(batch)
        return batch, index_target

    def step(self, batch, step_label):
        batch, index_target = self.prediction(batch)
        evals, parametric_flow = self.Criterion(batch)
        if index_target is not None:
            batch['Pred'] = batch['PredV'][:,:,index_target]
            batch['ParametricFlow'] = parametric_flow[:,:,index_target]
        self.Evaluations(batch, evals) # Add evaluations to evals dict
        self.log_dict({f'{step_label}/{k}':v.item() for k, v in evals.items() if v.dim() == 0},\
                      on_epoch=True, on_step=False, batch_size=1)
        return evals, batch

    def training_step(self, batch, batch_idx) :
        evals, batch =  self.step(batch, 'train')
        return evals

    def validation_step(self, batch, batch_idx) :
        evals, batch =  self.step(batch, 'val')
        return evals

    def test_step(self, batch, batch_idx) :
        evals, batch =  self.step(batch, 'test')
        evals.update({"batch":dict([(k, v.cpu()) if torch.is_tensor(v) else (k, v) for k,v in batch.items()])}) # For displaying we include the batch in the output
        return evals

    def Evaluations(self, batch, evals) :
        """
        Run evaluations of the binary masks and add the key to evals
        Args : batch with at least keys
                    'Pred' (b, L, I, J) : Mask proba predictions with sofmax over dim=1
                    'GtMask' binary (b, I, J) : ground truth foreground mask
               evals with a least keys
                    'losses' (b) : Loss per image
        """
        assert  batch['Pred'].sum(axis=1).allclose(torch.tensor([1.0], device=batch['Pred'].device)),\
               'Prediction must sum to 1 for the masks'

        self.generate_binary_mask(batch) # Generate Binary Maks
        evals['loss'] = evals['losses'].mean() # Loss is necessary for backprop in Pytorch Lightning
        a = batch['Pred'].argmax(axis=1).flatten(1).detach()
        evals['masks_usages'] = torch.tensor([len(a[i].unique()) for i in range(a.shape[0])])
        evals['mask_usage'] = evals['masks_usages'].to(torch.float).mean()
        if 'GtMask' in batch :
            evals['jaccs'] = multi_db_eval_iou(batch['GtMask'], batch['PredMask'].to(int))
            evals['jacc'] = evals['jaccs'].mean()
        evals['coherence_loss'] = evals['coherence_losses'].mean()
        evals['regularisation_loss'] = evals['regularisation_losses'].mean()
        for k in self.regularisation.keys() :
            evals[k+'_avg'] = evals[k].mean()

    def generate_result_fig(self, batch, evals) :
        """
        Compute the results for the given batch using th model and generate
        a figure presenting the results
        batch : images and modalities
        evals : evaluation metrics dict
        """
        sh = lambda x : flowiz.convert_from_flow(x.detach().cpu().permute(1,2,0).numpy())
        with torch.no_grad() :
            fig, ax = plt.subplots(2, 3, figsize = (15,10))
            if 'Image' in batch.keys() :
                ax[0,0].set_title('Image')
                ax[0,0].imshow((batch['Image'].permute(1,2,0) * 0.2 )+0.5) # By default the image is normalised between [-0.5;0.5]
            if 'Flow' in batch.keys() :
                ax[0,1].set_title(f'Flow min: {torch.min(batch["Flow"]):.2f} max: {torch.max(batch["Flow"]):.2f}')
                ax[0,1].imshow(sh(batch['Flow']))
            ax[1,0].set_title(f'Pred : {evals["losses"]:.3f}')
            ax[1,0].imshow(batch['Pred'].argmax(0), vmin=0, vmax=batch['Pred'].shape[0]) # For BCE Net pred 1 is object
            ax[1,1].set_title('PredMask')
            ax[1,1].imshow(batch['PredMask'], vmin=0, vmax=batch['GtMask'].max().item())
            if 'GtMask' in batch.keys() :
                ax[1,2].set_title(f'GtMask {evals["jaccs"]:.3f}')
                ax[1,2].imshow(batch['GtMask'], vmin=0, vmax=batch['GtMask'].max().item())
            else :
                ax[1,2].set_title(f'Confidence (max proba)')
                ax[1,2].imshow(batch['Pred'].max(axis=0).values)

            ax = self.custom_figs(ax, batch, evals) # Add axes corresponding to the class
        return fig

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = LitBackbone.add_specific_args(parser)
        parser.add_argument('--nAffineMasks', '-L', type=int, default=2)
        parser.add_argument('--model_type', '-mt', type=str, choices=['coherence_B'], default='coherence_B')
        parser.add_argument('--optim.name', type=str, choices=['Adam'], default='Adam')
        parser.add_argument('--optim.schedule', type=str, choices=['StepLR', 'CyclicLR', 'WarmupDecay'], default='')
        parser.add_argument('--optim.lr', type=float, default=1e-2)
        parser.add_argument('--optim.weight_decay', type=float, default=0)
        parser.add_argument('--finetune', type=str, default='none')
        return parser
