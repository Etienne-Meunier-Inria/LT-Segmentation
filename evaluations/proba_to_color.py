import sys; from __init__ import PRP; sys.path.append(PRP)

import os, torch, numpy as np,  pandas as pd, sys
from pathlib import Path
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from PIL import Image
from argparse import ArgumentParser
from tqdm import tqdm
from ipdb import set_trace

from evaluations.utils_evals import SequenceDataloader
from evaluations.evaluation import get_sequence


def save_argmax(mat, path) :
    """
    Save an indexed image represented by a matrix to the specified path.

    Parameters:
        mat : np.ndarray
            Matrix representing the indexed image.
        path : str
            Path to save the indexed image.
    """
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    mat = np.array(mat).astype('uint8')
    save_indexed(path, mat)

# From https://github.com/Jyxarthur/OCLR_model/blob/89ad2339107368ae5e4e23479a1605088170354d/utils.py#L120
def imwrite_indexed(filename, array, colour_palette):
    """
    Save an indexed PNG image using the specified filename, array, and color palette.

    Parameters:
        filename : str
            Path to save the PNG image.
        array : np.ndarray
            Array representing the indexed image.
        colour_palette : np.ndarray
            Color palette for the image.
    """
    # Save indexed png for DAVIS
    im = Image.fromarray(array)
    im.putpalette(colour_palette.ravel())
    im.save(filename, format='PNG')

def save_indexed(filename, img, colours = [[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]]):
    """
    Save an indexed image using the specified filename, image matrix, and color palette.

    Parameters:
        filename : str
            Path to save the indexed image.
        img : np.ndarray
            Matrix representing the indexed image.
        colours : list, optional (default=[[128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128]])
            Color palette for the image.
    """
    colour_palette = np.array([[0,0,0]] + colours).astype(np.uint8)
    imwrite_indexed(filename, img, colour_palette)



def main(model_dir, data_file, base_dir, cs, duplicate_last) :
    """
    Parameters:
        model_dir : str
            Directory containing the segmentation model.
        data_file : str
            Data file to evaluate the model on.
        base_dir : str
            Base directory for the dataset.
        cs : int or None
            Cut size for processing temporal sequences. If None, the entire sequence is processed in one step.
        duplicate_last : bool
            Duplicate last frame with an idented file name (for Davis17 prediction)

    """
    sl = SequenceDataloader('DataSplit/'+data_file+'_{}.csv',
                                img_size_gen=None,
                                request_fields=['GtMask', 'Flow'],
                                base_dir=base_dir)
    for split in ['val', 'train'] :
        for batch in tqdm(sl.get_sequence_loader(split)) :
            suffix=f'_cs={cs}'
            seq, gt_masks, seq_path, gt_paths  = get_sequence(batch, model_dir, suffix)
            for i, gt_path in enumerate(gt_paths) :
                if gt_path is not None :
                    save_path = model_dir + '/'+ gt_path.replace('Annotations', f'Results{suffix}')
                    save_argmax(seq[:,i].argmax(0) +1, save_path)
            if duplicate_last :
                pl = Path(save_path)
                last_path =  pl.parent / f"{int(pl.stem) + 1:0{len(pl.stem)}d}{pl.suffix}"
                save_argmax(seq[:,i].argmax(0) +1, last_path)


if __name__ == '__main__' :
    parser = ArgumentParser()
    parser.add_argument('--model_dir', '-md', type=str)
    parser.add_argument('--base_dir', '-bd', type=str)
    parser.add_argument('--cut_size', '-cs', type=int, default=None)
    parser.add_argument('--data_file', type=str, nargs='+')
    parser.add_argument('--duplicate_last', action='store_true')


    args = parser.parse_args()
    if args.cut_size == -1 : args.cut_size = None
    for data_file in args.data_file :
        main(args.model_dir, data_file, args.base_dir, args.cut_size, args.duplicate_last)
