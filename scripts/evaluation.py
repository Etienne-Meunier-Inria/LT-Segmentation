import sys; from __init__ import PRP; sys.path.append(PRP)

import os, torch, numpy as np,  pandas as pd, sys

from tqdm import tqdm
from argparse import ArgumentParser
from scripts.utils_evals import metric_grid, relabel, selectmasks_topiou, SELECT_MASK, SequenceDataloader, cut_cs, stitch_cs
from pathlib import Path
from ShapeChecker import ShapeCheck
from scipy.optimize import linear_sum_assignment
from evaluations import multi_db_eval_iou, db_eval_iou
from ipdb import set_trace
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize

def get_sequence(batch, model_dir, suffix, return_flos=False, max_len=1000) :
    """
    Load sequence associated to a batch and model dir saved
    Resize sequence if needed.
    Args :
        batch (dict) : with keys ['Flow', ... 'Flow+T'] and ['GtMask' ... 'GtMask+T']
        model (str) : path of the model dir
    Return
        seq (torch.tensor) : (C, T, I, J) prediction for the full sequence
        gt_masks (torch.tensor) : (T, I, J) gt labels
        seq_path (str) : path of the sequence
        gt_paths (list str) : (T)
    """
    gts = ['GtMask'] + [f'GtMask+{i}' for i in range(1, max_len)]
    flos_r = ['Flow'] + [f'Flow+{i}' for i in range(1, max_len)]
    gts_p = ['GtMaskPath'] + [f'GtMask+{i}Path' for i in range(1, max_len)]
    gt_masks = [batch[gts[i]] for i in range(max_len) if batch[flos_r[i]] is not None]
    flos = torch.stack([batch[flos_r[i]] for i in range(max_len) if batch[flos_r[i]] is not None])
    assert gt_masks[0] is not None, 'gt_mask [0] is None'
    gt_masks = torch.stack([g if g is not None else torch.full(gt_masks[0].shape, -1) for g in gt_masks])
    gt_paths = [batch.get(gts_p[i], None) for i in range(max_len) if batch[flos_r[i]] is not None]

    sc = ShapeCheck(gt_masks.shape, 't i j')

    p = Path(model_dir) / Path(batch['FlowPath']).parent
    seq_path = f'{str(p)}/{p.name}{suffix}.npy'
    seq = torch.tensor(np.load(seq_path))
    if seq.shape[1] > max_len :
        seq = seq[:, :max_len]
    if seq.shape[-2:] != gt_masks.shape[-2:] :
        print('Warning : Resizing prediction to fit gt shape')
        seq = resize(seq, gt_masks.shape[-2:], interpolation=InterpolationMode.BILINEAR)
    sc.update(seq.shape[-3:], 't i j')
    if return_flos :
        return seq, gt_masks, seq_path, gt_paths, flos
    return seq, gt_masks, seq_path, gt_paths


def format_results(metric, labels_gt, index_metric_pred, index_metric_gt, gt_paths, exclude_minus_1_gt=True) :
    """
    Write results csv in matrix. Excluding {0}
    If a label is in gt but not pred then iou = 0
    Args :
        metric (B, labels_pred, labels_gt) : metric with all iou
        labels_gt (B, labels_gt) : list of labels gt for each frame in metric
        index_metric_pred (labels_pred) : list of labels pred ordered as in metric
        index_metric_gt (labels_gt) : list of labels gt pred ordered as in metric
        gt_paths (B) : List of gt paths treated
        exclude_minus_1_gt (bool) : Exclude sites with -1 label in gt indicating missing labels.
    Return :
        dataframe with Path and results for each pred
    """
    assert metric.shape[0] == len(labels_gt), 'Error in format'
    exclude = set()
    if exclude_minus_1_gt :
        exclude.add(-1)

    global_lgt = sorted(set(sum(labels_gt, []))- exclude)
    d = np.full((metric.shape[0], len(global_lgt)), np.nan)
    d = pd.DataFrame(d, columns=[f'pred_{lp}' for lp in global_lgt])

    for i in range(len(metric)) :
        d.loc[i, 'Path'] = gt_paths[i]
        labels_gt_i = labels_gt[i]
        for lgt in labels_gt_i :
            if lgt not in exclude :
                if lgt in index_metric_pred :
                    index = list(index_metric_pred).index(lgt)
                    index_gt = list(index_metric_gt).index(lgt)
                    d.loc[i, f'pred_{lgt}'] = metric[i, index, index_gt].item()
                else :
                    d.loc[i, f'pred_{lgt}'] =  0
    print(pd.DataFrame(d)['pred_1'].mean())
    return pd.DataFrame(d)

def main(model_dir, data_file, snm, cs, cst, img_size_gen=None) :
    sl = SequenceDataloader('DataSplit/'+data_file+'_{}.csv',
                            img_size_gen=img_size_gen,
                            request_fields=['Flow','GtMask'])
    #set_trace()
    select_mask = SELECT_MASK[snm]
    print(f'Evaluation with method {snm} cs : {cs} cst : {cst}')
    for split in ['val'] :
        d = []
        for batch in tqdm(sl.get_sequence_loader(split)) :
            seq, gt_masks, seq_path, gt_paths  = get_sequence(batch, model_dir, f'_cs={cs}')
            argmax_seq =  seq.argmax(0)

            argmax_seq, k , p = cut_cs(argmax_seq[None], cst)
            gt_masks_seq_l, k , p = cut_cs(gt_masks[None], cst)

            sc = ShapeCheck(argmax_seq.shape, 'b t i j')
            argmax_seq = sc.rearrange(argmax_seq, 'b t i j -> b (t i) j')
            gt_masks_seq = sc.rearrange(gt_masks_seq_l, 'b t i j -> b (t i) j')

            new_seq = select_mask(argmax_seq, gt_masks_seq)
            new_seq =  sc.rearrange(new_seq, 'b (t i) j -> b t i j')


            new_seq = stitch_cs(new_seq, cst, k, p)
            assert new_seq.shape[0] == 1, f'Error in pred shape : {new_seq.shape}'

            labels_gt_l = sum([[gtm.flatten().unique().tolist()]*gtm.shape[0] for gtm in gt_masks_seq_l],  [])
            if (p is not None) and (p > 0) : labels_gt_l = labels_gt_l[:-p]

            np.save(seq_path.replace(f'_cs={cs}', f'_slm_{snm}_cs={cs}_cst={cst}'), new_seq[0].numpy())
            metric, labels_pred, labels_gt = metric_grid(new_seq[0], gt_masks, db_eval_iou)
            ft = [g is not None for g in gt_paths]

            d.append(format_results(metric[ft],
                                    [l for l, f in zip(labels_gt_l, ft) if f],
                                    labels_pred, labels_gt,
                                    np.array(gt_paths)[ft]))
        df = pd.concat(d)
        df['Sequence'] = df['Path'].apply(lambda x : x.split('/')[-2])
        df.to_csv(f'{model_dir}/{data_file}_slm_{snm}_cs={cs}_cst={cst}_{split}.csv')
        print(split, df['pred_1'].mean())



if __name__ == '__main__' :
    parser = ArgumentParser()
    parser.add_argument('--model_dir', '-md', type=str)
    parser.add_argument('--select_mask', '-snm', type=str,
                        choices=['topiou', 'optimax', 'optimax_all', 'linear_assignment', 'exceptbiggest'])
    parser.add_argument('--cut_size', '-cs', type=int, default=None)
    parser.add_argument('--cut_size_select', '-cst', type=int, default=None)
    parser.add_argument('--data_file', type=str, choices=['DAVIS_D16Split',
                                                          'FBMSclean_FBMSSplit',
                                                          'SegTrackv2_EvalSplit'], nargs='+')
    args = parser.parse_args()
    if args.cut_size == -1 : args.cut_size = None
    if args.cut_size_select == -1 : args.cut_size_select = None
    for data_file in args.data_file :
        main(args.model_dir, data_file, args.select_mask, args.cut_size, args.cut_size_select)
