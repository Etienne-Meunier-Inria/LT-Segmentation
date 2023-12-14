import sys; from __init__ import PRP; sys.path.append(PRP)

from ShapeChecker import ShapeCheck
import torch, einops,pandas as pd
from evaluations import db_eval_iou
from scipy.optimize import linear_sum_assignment
from ipdb import set_trace

import os, torch, einops, math, numpy as np, yaml
from glob import glob
from csvflowdatamodule.CsvDataset import CsvDataModule
from Models.BCENet import BCENet
from Models.CoherenceNets.MethodeB import MethodeB
from torch.nn.functional import pad
import itertools

def get_name_lowest_loss_ckpt(model_dir) :
    lpm = glob(os.path.join(model_dir,'checkpoints/*epoch_val_loss*.ckpt'))
    pm = sorted(lpm, key = lambda x : float(x.split(':')[-1].strip('.ckpt')))[0]
    return pm

def prepare_model(model_dir, img_size_gen=None,  binary_method_gen=None,
                  len_seq_gen=None, last_model=False, name_ckpt=None,extra_hparams={}) :
    """
    Load the best checkpoint of this model ( based on the validation loss)
    Args :
        model_dir (str) : path of the model to load containing ckpt files
        img_size_gen (list int) : size of the image to generate
        len_seq (int) : lenght of the sequence ( temporal dimension)
        last_model (bool) : if last_model is true then load the last weights instead of the lowest val loss
        extra_hparams (dict) : dictionnary of hparams to replace the original values
    Returns :
        net (nn.Module) : return loaded pytorch models
    """
    if name_ckpt is not None :
        pm = os.path.join(model_dir,'checkpoints/', name_ckpt)
    elif last_model :
        pm = os.path.join(model_dir,'checkpoints/last.ckpt')
    else :
        pm = get_name_lowest_loss_ckpt(model_dir)
    print(f'Using Checkpoint : {pm}')

    t = torch.load(pm, map_location='cpu')
    print('Model type : ', t['hyper_parameters']["backbone"])
    if binary_method_gen is None :
        binary_method_gen = t['hyper_parameters']['binary_method']
    if t['hyper_parameters']['model_type'] == 'coherence_B' :
        net = MethodeB.load_from_checkpoint(pm, strict=False, binary_method=binary_method_gen, **extra_hparams)
    elif t['hyper_parameters']['model_type'] == 'bce' :
        net = BCENet.load_from_checkpoint(pm, strict=False, binary_method=binary_method_gen, **extra_hparams)
    print(f'Hyperparameters : {net.hparams}')
    with open(pm.replace('ckpt', 'hyp'), 'w') as f : yaml.dump(net.hparams, f)

    if img_size_gen is not None and len_seq_gen is not None:
        print(f'Change Size Gen : {img_size_gen} {len_seq_gen}')
        net.init_crit(img_size_gen, len_seq_gen, net.hparams['param_model'])
        net.hparams['img_size'] = img_size_gen


    net.data_path  = f'DataSplit/{net.hparams.data_file}_'+'{}.csv'
    return net


def make_grid(array, col_width=10) :
    """
    Draw and save a grid of the temporal array where the frames
    are ordered in the lexical order.
    Grid will have final shape (up(t//col_width), col_width)
    Args :
        array (t, i, j) : array to turn to grid
        path_save : path to save the image in
        col_width : col_width of the grid
    Return :
        grid (up(t//col_width)*i, col_width*j) : returned as an array
    """
    if (array.shape[0] % col_width) != 0 :
        p = col_width - (array.shape[0] % col_width)
        array = np.pad(array, ((0,p), (0,0), (0,0)), constant_values=0)
    im = einops.rearrange(array, '(k h) i j -> (k i) (h j)', h = col_width)
    return im


class SequenceDataloader :

    def __init__(self, data_path, img_size_gen, request_fields=['Flow'], base_dir=os.environ['Dataria'], max_len=1000) :
        request = request_fields + [f'{rqi}+{i}' for rqi in request_fields for i in range(1, max_len)]
        d = {'preload_cache' : False,
             'img_size' : img_size_gen,
             'batch_size' :1,
             'base_dir' :base_dir,
             'flow_augmentation' : '',
             'image_augmentation' : '',
             'mix_augmentation' : '',
             'shuffle_fit' :False,
             'boundaries':'Ignore'}
        self.dm = CsvDataModule(data_path =data_path, request=request,\
                   num_workers=8, **d)
        self.dm.setup('fit')

    def get_dataset(self, split) :
        if split == 'train' :
            dataset = self.dm.dtrain
        elif split == 'val' :
            dataset = self.dm.dval
        dataset.lenient_loader = True
        return dataset

    def get_sequence_starts(self, dataset) :
        """
        Get dict with all sequences starts
        Return :
            sequences_start : {sequence : index_start}
        """
        sequences = [(k, v[0], v[1]) for k,v in dataset.fs.idx_map.items()]
        df = pd.DataFrame(sequences, columns=['Index', 'Sequence', 'Seq_index'])
        sequences_start = df.sort_values('Seq_index').groupby('Sequence').first()['Index'].to_dict()
        return sequences_start

    def get_sequence_loader(self, split) :
        dataset =  self.get_dataset(split)

        #sequences_start = [k for k,v in dataset.fs.idx_map.items() if v[1] == 0]
        sequences_start = self.get_sequence_starts(dataset)
        return (dataset.__getitem__(sqt) for sqt in sequences_start.values())

    def get_sequence(self, sequence) :
        in_train = sequence in set([v[0] for k,v in self.dm.dtrain.fs.idx_map.items()])
        in_val = sequence in set([v[0] for k,v in self.dm.dval.fs.idx_map.items()])
        assert not (in_train and in_val), f'Sequence {sequence} in both splits !'
        assert (in_train or in_val), f'Sequence {sequence} missing'
        assert in_train ^ in_val, f'Error ! {sequence}'
        split = 'train' if in_train else 'val'

        dataset =  self.get_dataset(split)
        sequences_start = self.get_sequence_starts(dataset)
        return dataset.__getitem__(sequences_start[sequence])

def cut_cs(array, cs) :
    """
    Cut an array on the temporal dimension into minibatch of time cs
    First pad the array using replicate to fill the last batch then
    proceed to cut.
    1 . (b, ..., t, i, j) -> (b, ..., up(t/cs)*cs, i, j)
    2. (b, ..., up(t/cs)*cs, i, j) -> (b*up(t/cs), ..., cs, i, j)
    Args :
        array : (b, ..., t, i, j)
        cs (int) : size of the cuts
    Returns :
        cutted array : (b*up(t/cs), ..., cs, i, j)
        k : up(t/cs), number of cuts
        p : up(t/cs)*cs - t, padding added at the end of the t dim
    """
    if cs is None : return array, None, None
    t =  array.shape[-3]
    if cs > t : return array, None, None
    p = math.ceil(t/cs) * cs - t
    assert p >= 0, 'Error padding'
    array =pad(array.to(torch.float64), (0,0,0,0,0,p), mode='replicate').to(array.dtype)
    k = array.shape[-3] // cs
    array = einops.rearrange(array, 'b ... (k cs) i j -> (b k) ... cs i j', cs=cs, k=k)
    return array, k, p

def stitch_cs(array, cs, k, p) :
    """
    Stich back the pieces of a previously cut array
    Args :
        array : cutted array (b*up(t/cs), ..., cs, i, j)
        cs : cut size used for the cut
        k : number of cuts made
        p : padding added
    Return
        new_array : (b, ..., t, i, j)  after stiching
    """
    if k is None :
        t = array.shape[-3]
        assert (cs is None) or (cs > t), f'Error in stitch_cs t: {t} cs:{cs} k:{k}'
        return array
    ar = einops.rearrange(array, '(b k) ... cs i j -> b ... (k cs) i j', cs=cs, k=k)
    if p > 0 : ar = ar[..., :-p, :, :]
    return ar

def metric_grid(argmax_pred, gt_mask, function, exclude_minus_1_gt=True) :
    """
    Compare the masks for each labels between the predictions
    and the ground truth. Use function to compare binary masks
    and return a grid with the comparison for all labels.
    Args :
        argmax_pred (b, i ,j) : prediction argmax with labels in [labels_pred]
        gt_mask (b, i, j) : gt mask with labels in [labels_gt]
        function (b, i, j)*(b, i, j) -> R: function that takes two binary mask and return a metric
        exclude_minus_1_gt (bool) : Exclude sites with -1 label in gt indicating missing labels.
    Return :
        metric (b, labels_pred, labels_gt) : grid with the comparison for all pairs of labels
        labels_pred [int]: labels in pred in ame order as in metric (row names)
        labels_gt [int]: labels in gt in same order as in metric (col names)
    """
    sc = ShapeCheck(argmax_pred.shape, 'b i j')
    sc.update(gt_mask.shape, 'b i j')
    labels_pred, labels_gt = argmax_pred.unique(), gt_mask.unique()
    if exclude_minus_1_gt and (-1 in gt_mask):
        print(f'METRIC GRID : Exclude gt label -1')
        argmax_pred = (gt_mask == -1) * -1 + (gt_mask != -1) * argmax_pred
    metric = torch.zeros((sc.get('b')['b'], len(labels_pred), len(labels_gt)))
    for i, li in enumerate(labels_pred) :
        for j, lj in enumerate(labels_gt) :
            metric[:, i, j] = function(argmax_pred==li, gt_mask==lj)
    return metric, labels_pred, labels_gt



def relabel(argmax_pred, mapping, labels_pred)  :
    """
    Change the labels in argmax_pred depending on the provided mapping.
    Args :
        argmax_pred (b, i ,j) : prediction argmax with labels in [labels_pred]
        mapping (b, labels_pred) : gives for each batch and labels pred the index to set
                ex : mapping[0] = [10,12,12,5]
                all labels in element 0 of the batch are modified with :
                    labels_pred[0] -> 10
                    labels_pred[1] -> 12
                    labels_pred[2] -> 12
                    labels_pred[3] -> 5
        labels_pred [int]: labels in pred in same order as in metric (row names)
    Return :
        new_seq (b, i, j) : prediction labels after mapping
    """
    assert set(argmax_pred.unique().numpy()) == set(labels_pred.numpy()), 'Error in labels in relabels'
    new_seq = torch.zeros_like(argmax_pred)
    for i, li in enumerate(labels_pred) :
        new_seq += (argmax_pred == li) *  mapping[:, i][:, None, None]
    return new_seq


def selectmasks_topiou(argmax_seq, gt_masks) :
    """
    Relabel the argmax_seq prediction using the gtmasks
    each predicted label take the label of the gt mask with which
    it has the best iou.
    Operation is done independently on each batch
    Warning : Using Gt for evaluation
    Warning : Several pred can be associated to one GT
    Args :
        argmax_seq (torch.tensor) : (B, I, J) argmax predictions
        gt_masks (torch.tensor) : (B, I, J) gt labels
    Returns :
        new_seq ( torch.tensor) : (B, I, J) labels after relabeling
    """
    sc = ShapeCheck(argmax_seq.shape, 'b i j')
    sc.update(gt_masks.shape, 'b i j')
    metric, labels_pred, labels_gt = metric_grid(argmax_seq, gt_masks, db_eval_iou)
    sc.update(labels_pred.shape, 'n_lb_pred')
    sc.update(labels_gt.shape, 'n_lb_gt')
    sc.update(metric.shape, 'b n_lb_pred n_lb_gt')
    mapping = labels_gt[metric.argmax(-1)]
    print(mapping)
    new_seq = relabel(argmax_seq, mapping, labels_pred)
    sc.update(new_seq.shape, 'b i j')
    return new_seq


def selectmasks_exceptbiggest(argmax_seq, *args) :
    """
    Relabel the argmax_seq prediction by labelling all segments a 1 except the biggest one.
    Operation is done independently on each batch
    Args :
        argmax_seq (torch.tensor) : (B, I, J) argmax predictions
    Returns :
        new_seq ( torch.tensor) : (B, I, J) labels after relabeling
    """
    sc = ShapeCheck(argmax_seq.shape, 'b i j')
    modes = argmax_seq.flatten(1,2).mode().values
    labels_pred = argmax_seq.unique()
    labels_pred_list =  list(labels_pred)
    mapping = torch.ones(argmax_seq.shape[0], len(labels_pred), dtype=labels_pred.dtype)
    for i, mode in enumerate(modes) :
        mapping[i, labels_pred_list.index(mode)] = 0

    sc.update(labels_pred.shape, 'n_lb_pred')

    new_seq = relabel(argmax_seq, mapping, labels_pred)
    sc.update(new_seq.shape, 'b i j')
    return new_seq

def selectmasks_linear_sum_assignment(argmax_seq, gt_masks, exclude_background=False) :
    """
    Relabel the argmax_seq prediction using the gtmasks
    each gt mask label is associated to a predicted label
    to maximise the final iou.
    Operation is done independently on each batch
    Warning : Using Gt for evaluation
    Warning : Only one pred is associated to each GT
    Args :
        argmax_seq (torch.tensor) : (B, I, J) argmax predictions
        gt_masks (torch.tensor) : (B, I, J) gt labels
    Returns :
        new_seq ( torch.tensor) : (B, I, J) labels after relabeling
    """
    sc = ShapeCheck(argmax_seq.shape, 'b i j')
    sc.update(gt_masks.shape, 'b i j')

    # Compute Grid
    metric, labels_pred, labels_gt = metric_grid(argmax_seq, gt_masks, db_eval_iou)
    sc.update(labels_pred.shape, 'n_lb_pred')
    sc.update(labels_gt.shape, 'n_lb_gt')
    sc.update(metric.shape, 'b n_lb_pred n_lb_gt')

    if exclude_background :
        print('Linear Assignment : exclude background')
        # linear_assignment ( Excluding the label 0 as in D17)
        filter = (labels_gt != 0)
        labels_gt = labels_gt[filter]
        metric =  metric[:, :, filter]

    batch = metric.shape[0]
    mapping = torch.zeros(batch, len(labels_pred), dtype=int)
    for i in range(batch) :
        row_index, col_index = linear_sum_assignment(-metric[i])
        mapping[i][row_index] = labels_gt[col_index]

    # Relabel
    new_seq = relabel(argmax_seq, mapping, labels_pred)
    sc.update(new_seq.shape, 'b i j')
    return new_seq


def selectmasks_optimax_argmax(argmax_seq, gt_masks, max_segs=100, exclude_minus_1_gt=True) :
    """
    Choose the best labels based on the comparison with gt_masks based on time t only ( central time)
    Params :
        argmax_seq (b i j) argmax map with times [t-k... t ... t+k] with coherent segmentation labels
        gt_masks (b i j) gt map for time t
    Returns
        optimax_masks_vol (b i j) binary labels with the best selection
    """
    sc = ShapeCheck(argmax_seq.shape, 'b i j')
    sc.update(gt_masks.shape, 'b i j')

    if exclude_minus_1_gt and (-1 in gt_masks):
        assert set(gt_masks.unique().numpy()).issubset({-1,0,1}), 'gt_masks not binary'
        assert (argmax_seq >= -1).all(), 'Error in argmax_seq'

    else :
        assert set(gt_masks.unique().numpy()).issubset({0,1}), 'gt_masks not binary'
        assert (argmax_seq >= 0).all(), 'Error in argmax_seq'


    assert not torch.is_floating_point(argmax_seq), 'Error in argmax_seq'

    L = argmax_seq.max().item() +1
    sc.update([L], 'L')

    binmax = sc.repeat(torch.zeros(1, device=argmax_seq.device), '1 -> b L i j').clone()
    binmax.scatter_(1, argmax_seq[:, None], 1)

    if exclude_minus_1_gt and (-1 in gt_masks):
        print(f'METRIC GRID : Exclude gt label -1')
        binmax[(gt_masks==-1)[:,None].repeat(1,binmax.shape[1],1,1)] = 0.0
    s_channels = []
    for itr in range(min(L, max_segs)) :
        bench=[]
        for k in range(L) :
            pdm = sum([binmax[torch.arange(sc.get('b')['b']),s,:,:] == 1.0
                                     for s in s_channels + [k]]).to(torch.bool)
            bench.append(db_eval_iou(gt_masks==1, pdm==True))
        bench = torch.stack(bench)
        s_channels.append(bench.argmax(axis=0))
    optimax_masks = sum([argmax_seq == sc.repeat(s, 'b -> b i j') for s in s_channels]).to(bool)
    return optimax_masks

def selectmasks_optimax_all(argmax_seq, gt_masks, max_segs=100, exclude_minus_1_gt=True) :
    """
    Choose the best labels based on the comparison with gt_masks based on time t only ( central time)
    Params :
        argmax_seq (b i j) argmax map with times [t-k... t ... t+k] with coherent segmentation labels
        gt_masks (b i j) gt map for time t
    Returns
        optimax_masks_vol (b i j) binary labels with the best selection
    """
    sc = ShapeCheck(argmax_seq.shape, 'b i j')
    sc.update(gt_masks.shape, 'b i j')
    if exclude_minus_1_gt and (-1 in gt_masks):
        assert set(gt_masks.unique().numpy()).issubset({-1,0,1}), 'gt_masks not binary'
        assert (argmax_seq >= -1).all(), 'Error in argmax_seq'

    else :
        assert set(gt_masks.unique().numpy()).issubset({0,1}), 'gt_masks not binary'
        assert (argmax_seq >= 0).all(), 'Error in argmax_seq'


    assert not torch.is_floating_point(argmax_seq), 'Error in argmax_seq'

    # Compute Grid
    intersection = lambda x, y : (x.to(torch.bool) & y.to(torch.bool)).sum(axis=(1,2))
    union = lambda x, y : (x.to(torch.bool) | y.to(torch.bool)).sum(axis=(1,2))
    metric_inter, labels_pred, labels_gt = metric_grid(argmax_seq, gt_masks, intersection)
    metric_union, labels_pred_u, labels_gt_u = metric_grid(argmax_seq, gt_masks, union)
    assert labels_pred.equal(labels_pred_u) and labels_gt_u.equal(labels_gt_u), 'Error in metric'
    sc.update(labels_pred.shape, 'n_lb_pred')
    sc.update(labels_gt.shape, 'n_lb_gt')
    sc.update(metric_inter.shape, 'b n_lb_pred n_lb_gt')
    sizes_gt = torch.stack([(gt_masks==i).sum(axis=(-1,-2)) for i in labels_gt], -1).to(torch.float32)


    L = len(labels_pred)
    combinations = torch.stack([torch.FloatTensor(i) for i in itertools.product([0, 1], repeat = L)])

    gt_index =  list(labels_gt).index(1)

    inters = einops.einsum(metric_inter[:,:,gt_index], combinations, 'n k, c k -> n c')
    unions = einops.einsum(metric_union[:,:,gt_index], combinations, 'n k, c k -> n c')  - ((combinations.sum(axis=-1)[None]-1) * sizes_gt[:,gt_index][:,None])
    values, indices = (inters / unions).max(axis=1)

    new_seq = relabel(argmax_seq, combinations[indices].to(int), labels_pred)

    return new_seq



SELECT_MASK = {'topiou' : selectmasks_topiou,
               'linear_assignment' : selectmasks_linear_sum_assignment,
               'optimax' : selectmasks_optimax_argmax,
               'optimax_all' : selectmasks_optimax_all,
               'exceptbiggest' : selectmasks_exceptbiggest}
