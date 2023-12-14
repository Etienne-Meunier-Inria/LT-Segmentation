import sys; from __init__ import PRP; sys.path.append(PRP)

import sys, os, einops, torch.nn as nn, torch, numpy as np
from copy import deepcopy
from csvflowdatamodule.CsvDataset import CsvDataModule
from scripts.utils_evals import prepare_model, SequenceDataloader, cut_cs, stitch_cs
from pathlib import Path
from tqdm import tqdm
from argparse import ArgumentParser
from ipdb import set_trace
import matplotlib
from csvflowdatamodule.utils.KeyTypes import parse_request

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model(model_dir, last_model=False, blockNorm=False,**kwargs) :
    """
    Retrieve the trained model and its hyperparameters.

    Parameters
    ----------
    model_dir (str): Path to the directory containing the trained model.
    last_model (bool): If True, load the latest model checkpoint. If False, load the best model checkpoint.
    blockNorm (bool): If True, use cBlockNorm3d normalization.

    Returns
    -------
    backbone : torch.nn.Module
        The backbone of the loaded model.
    hparams : dict
        Hyperparameters of the loaded model.
    """
    extra_hparams  = {}
    if blockNorm :
        print('Warning : using cBlockNorm3d normalisation')
        extra_hparams = {'unet.inner_normalisation' : 'cBlockNorm3d'}
    model = prepare_model(model_dir, last_model=last_model, extra_hparams=extra_hparams).eval();
    model.hparams["flow_augmentation"] = ''
    model.to(device);
    return model.backbone_model.model.model, model.hparams

@torch.no_grad()
def compute_features_preds(backbone, batch, cs=None, return_auxs=False, low_memory=False) :
     """
    Compute features and predictions from the backbone model.

    Parameters
    ----------
    backbone : torch.nn.Module
        The backbone model for computation.
    batch : dict
        Batch containing the input flows and related information.
    cs : int or None
        Cut size for processing temporal sequences. If None, the entire sequence is processed in one step.
    return_auxs : bool
        If True, return auxiliary information along with predictions.
    low_memory : bool
        If True, use low memory mode for computation.

    Returns
    -------
    preds : torch.Tensor
        Predictions for the input batch.
    auxs : torch.Tensor or None
        Auxiliary information if return_auxs is True, otherwise None.
    """
    inputs = ['Flow'] + [f'{Flow}+{i}' for i in range(1, 1000)]
    flowv = torch.stack([batch[k] for k in inputs if batch[k] is not None], dim=1)
    flowv, k , p = cut_cs(flowv[None], cs)
    if low_memory :
        assert not return_auxs, 'Low memory and return auxs incompatible'
        print('Low memory mode')
        preds =  []
        for flowvi in flowv :
            preds_i, _ = backbone(flowvi[None].to(device))
            preds.append(preds_i.to('cpu'))
        preds = torch.cat(preds)
    else :
        preds, auxs = backbone(flowv.to(device))
    preds = torch.softmax(preds, dim=1)
    preds = stitch_cs(preds, cs, k, p)
    assert preds.shape[0] == 1, f'Error in pred shape : {preds.shape}'
    if return_auxs :
        return preds[0], auxs
    return preds[0]

def save_preds(preds, batch, save_dir, cs=None) :
       """
    Save predictions to a numpy file.

    Parameters
    ----------
    preds : torch.Tensor
        Predictions to be saved.
    batch : dict
        Batch containing information about the flow path.
    save_dir : str
        Directory to save the predictions.
    cs : int or None
        Cut size used for naming the saved file.
    """

    p = Path(save_dir) / Path(batch[f'FlowPath']).parent
    p.mkdir(parents=True, exist_ok=True)
    np.save(f'{str(p)}/{p.name}_cs={cs}.npy', preds.cpu().numpy())


def main(model_dir, data_file, cs, img_size=None, blockNorm=False) :
     """
    Main function to generate predictions

    Parameters
    ----------
    model_dir : str
        Path to the directory containing the trained model.
    data_file : str
        Name of the data file to process.
    cs : int or None
        Cut size for processing temporal sequences. If None, the entire sequence is processed in one step.
    img_size : list or None
        Image size for generation. If None, use the model's default img_size.
    blockNorm : bool
        If True, replace original normalization with cBlockNorm3d.
    """
    backbone, hparams = get_model(model_dir, last_model=False, blockNorm=blockNorm)
    sl = SequenceDataloader('DataSplit/'+data_file+'_{}.csv',
                            img_size_gen= img_size if img_size is not None else hparams['img_size'],
                            request_fields=['Flow'])
    for split in ['val'] :
        for batch in tqdm(sl.get_sequence_loader(split)) :
            print(batch[f'FlowPath'])
            preds = compute_features_preds(backbone, batch, cs, low_memory=(data_file in ['SegTrackv2_EvalSplit', 'FBMSclean_FBMSSplit']))

if __name__ == '__main__' :
    parser = ArgumentParser()
    parser.add_argument('--model_dir', '-md', type=str)
    parser.add_argument('--cut_size', '-cs', type=int, default=None, help='Length of the temporal sequence to process')
    parser.add_argument('--data_file', type=str, choices=['DAVIS_D16Split','DAVIS17_D17Split', 'FBMSclean_FBMSSplit',
                                                          'SegTrackv2_EvalSplit'], nargs='+')
    parser.add_argument('--img_size', nargs='+', type=int, default=None, help='Image size generation, Default : model default img_size')
    parser.add_argument('--blockNorm', action='store_true', help='replace original normalisation with cBlockNorm3d')
    args = parser.parse_args()
    if args.cut_size == -1 : args.cut_size = None
    for data_file in args.data_file :
        main(args.model_dir, data_file, args.cut_size, args.img_size, args.blockNorm)
