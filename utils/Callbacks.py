import pytorch_lightning as pl
import flowiz
import torch
import wandb
from ipdb import set_trace
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def batch_to(batch, device) :
    """
    Move the elements in the batch to the specified device.

    Parameters
    ----------
    batch : dict
        Batch containing elements to move to the device.
    device : str
        The device (e.g., 'cuda' or 'cpu').

    Returns
    -------
    batch : dict
        Batch with elements moved to the specified device.
    """
    for key, value in batch.items():
        if type(value) == torch.Tensor :
            batch[key] = batch[key].to(device)
    return batch

class ResultsLogger(pl.Callback) :
    """
    PyTorch Lightning Callback to log and save results during training.

    Parameters
    ----------
    keys : list
        List of keys to log in the results CSV file.
    filepath : str or None
        Path to the results CSV file. If None, a default path will be used.

    Methods
    -------
    setup(trainer, pl_module, stage)
        Setup method to initialize the callback.

    write_results(imps, outputs, epoch, step_label)
        Write results to the results CSV file.

    on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None)
        Callback method at the end of each training batch.

    on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None)
        Callback method at the end of each validation batch.

    on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None)
        Callback method at the end of each test batch.

    on_test_end(trainer, pl_module)
        Callback method at the end of the test phase.

    Attributes
    ----------
    fp : str
        Path to the results CSV file.
    keys : list
        List of keys to log in the results CSV file.
    summary_path : str
        Path to the summary TSV file.
    summary_log : pandas.DataFrame
        DataFrame containing summary statistics.
    """
    def __init__(self, filepath=None,  keys=['losses', 'jaccs', 'masks_usages']):
        super().__init__()
        self.fp = filepath
        self.keys = keys

    def setup(self, trainer, pl_module, stage) :
        if self.fp is None :
            self.fp = os.path.join(trainer.log_dir, trainer.logger.name, trainer.logger.experiment.id, 'results.csv')
        print(f'Save results in {self.fp}')
        with open(self.fp, 'w') as f :
            f.write(f'epoch,step_label,file_name,'+','.join(self.keys)+'\n')

    @torch.no_grad()
    def write_results(self, imps, outputs, epoch, step_label) :
        with open(self.fp, 'a') as f :
            for i, imn in enumerate(imps) :
                f.write(f'{epoch},{step_label},{imn},'+','.join([f'{outputs[j][i].item():.3f}' for j in self.keys if j in outputs.keys()])+'\n')

    def batch_end(self, trainer, outputs, batch, step_label):
        if batch is None : return None
        key_path = 'ImagePath' if 'ImagePath' in batch.keys() else 'FlowPath'
        self.write_results(batch[key_path], outputs, trainer.current_epoch, step_label)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        self.batch_end(trainer, outputs, batch, 'train')

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        self.batch_end(trainer, outputs, batch, 'val')

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=None):
        self.batch_end(trainer, outputs, batch, 'test')

    def on_test_end(self, trainer, pl_module) :
        self.summary_path = self.fp.replace('.csv', '_summary.tsv')
        dfr = pd.read_csv(self.fp)
        dfr['sequence'] = dfr['file_name'].apply(lambda x : x.split('/')[-2])
        dfr['dataset'] = dfr['file_name'].apply(lambda x : x.split('/')[0])
        dfr.groupby('sequence').mean().to_csv(self.summary_path, sep='\t')
        self.summary_log = dfr.mean()
        self.summary_log.to_csv(self.summary_path, sep='\t', mode='a', header=False)
        print(f'Summary saved at : {self.summary_path}')


class SaveResultsFig(pl.Callback) :
    """
    PyTorch Lightning Callback to save result images and probability maps.

    Parameters
    ----------
    save_dir : str
        Directory to save the images.

    Methods
    -------
    on_test_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        Callback method at the end of each test batch.

    extract_dict(dict, idx)
        Extract the index idx from the dictionary and push to the CPU if necessary.

    Attributes
    ----------
    save_dir : str
        Directory to save the images.
    """

    def __init__(self, save_dir) :
        self.save_dir = save_dir
        print(f'Saving Images in {self.save_dir}')

    @torch.no_grad()
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        batch_out = outputs.pop('batch')
        key_path =  'ImagePath' if 'ImagePath' in batch_out.keys() else 'FlowPath'
        for i in range(len(outputs['losses'])):
            batch_out_idx = self.extract_dict(batch_out, i)
            outputs_idx = self.extract_dict(outputs, i)
            p = Path(f'{self.save_dir}/{batch_out[key_path][i]}')
            p.parent.mkdir(parents=True, exist_ok=True)

            # Results Image representation
            fig = pl_module.generate_result_fig(batch_out_idx, outputs_idx)
            fig.suptitle(f'Run : {pl_module.hparams.experiment_name} ({pl_module.hparams.experiment_id}) \n'+
                         f'Image : {Path(batch[key_path][i])}', fontsize='xx-large')
            plt.tight_layout()
            fig.savefig(p.with_suffix('.png'))
            plt.close(fig)

            # Save probability masp
            np.save(p.parent / (p.stem +'_proba.npy'), batch_out_idx['PredV'].numpy())

    @staticmethod
    def extract_dict(dict, idx):
        """
        Extract the index idx from dict an push to cpu if necessary
        """
        r_dict = {}
        for k in dict.keys() :
            if torch.is_tensor(dict[k]) :
                if dict[k].dim() > 0 :
                    r_dict[k] = dict[k][idx].cpu()
                else :
                    r_dict[k] = dict[k].cpu()
            else : r_dict[k] = dict[k]
        return r_dict
