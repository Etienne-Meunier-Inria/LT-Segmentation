import torch
import numpy as np
#import cv2
from ipdb import set_trace

def db_eval_iou(annotation,segmentation):
    """
    Compute region similarity as the Jaccard Index.

    Arguments:
    annotation   (torch): binary annotation   map. (b, I, J)
    segmentation (ndarray): binary segmentation map. (b, I, J)

    Return:
    jaccard (float): region similarity (b)
    """
    annotation = annotation.to(torch.bool)
    segmentation = segmentation.to(torch.bool)

    k = (segmentation & annotation).sum(axis=(1,2)) / (segmentation | annotation).sum(axis=(1,2))
    k[k.isnan()] = 1 # 1 jaccard score if both gt and pred are empty
    return k

def multi_db_eval_iou(annotation, segmentation) :
    """
    Compute region similarity as the Jaccard Index with multimasks.
    Return the average of the IoU of all masks.
    Not counting Jacc for the background (=0) index.


    Arguments:
    annotation   (torch): binary annotation   map. (b, I, J) :int
    segmentation (torch): binary segmentation map. (b, I, J) : int

    Return:
    jaccard (float): region similarity (b)
    """
    assert segmentation.shape == annotation.shape, 'Error in shape between segmentation and annotation'
    assert segmentation.dtype == annotation.dtype == torch.int64, 'Error in input type'
    bs = annotation.shape[0]
    jaccs = torch.zeros(bs)
    for b in range(bs) :
        annot, seg = annotation[b], segmentation[b]
        idxs_gt = set(torch.unique(annot).tolist())
        assert set(torch.unique(seg).tolist()).issubset(idxs_gt),\
        f'Segmentation ({set(torch.unique(seg).tolist())}) presenting index not in annotation ({idxs_gt})'
        if (annot > 0).sum() == (seg > 0).sum() == 0 :
            print('Beware : empty mask in eval')
            jaccs[b] = 1.0
            continue
        idxs_gt.remove(0) # Not counting the backgrounds
        jaccs[b] = np.mean([db_eval_iou(annot[None] == idx, seg[None] == idx)[0].item() for idx in idxs_gt])
    return jaccs

def binarise_heuristic(mask) :
    """
    Convert a probability mask with two classes into a binary map where
    we have foreground=1, background=0. Based on the method from motion grouping
    (Zisserman, 2021) paper.

    Arguments :
    mask( torch) : probabilistic output mask (b, I, J) in [0, 1]

    Return:
    bmask (torch) : binary mask with foreground at 1 (b, I, J) in {0, 1}
    """
    _, h, w = mask.shape
    borders = 2*h+2*w-4
    n_wht_border = (mask > 0.5).sum(axis=(1,2)) - (mask[:, 1:-1, 1:-1]> 0.5).sum(axis=(1,2))
    scores = n_wht_border / borders
    mask[scores> 0.5] = 1 - mask[scores>0.5]

    thres = 0.1
    mask[mask > 0.1] = 1
    mask[mask <= 0.1] = 0
    return mask

def batch_bbox_mask(bmasks):
    """
    Call get_bbox_mask for a batch of torch tensors
    Arguments :
    bmasks (torch) : batch of binary mask with the foregroun at 1 : {0,1} (b, I, J)

    Returns :
    boxmasks (torch) : batch of binary mask with the foreground box at 1 : {0,1} ( b, I, J)
    """
    b = bmasks.shape[0]
    boxmasks = []
    for i in range(b) :
        boxmasks.append(get_bbox_mask(bmasks[0].numpy()))
    return torch.tensor(np.stack(boxmasks))

def get_bbox_mask(bmask) :
    """
    Get a binary mask ( one sample ) and extract a bounding box max with the bounding
    box around the biggest countour. Based on the method from motion grouping
    (Zisserman, 2021) paper.
    Arguments :
    bmask (ndarray) : binary mask with the foregroun at 1 : {0,1}

    Returns :
    boxmask (ndarray) : binary mask with the foreground box at 1 : {0,1}
    """
    bmask = (bmask * 255).astype(np.uint8)
    boxmask = np.zeros_like(bmask, dtype=bool)
    if bmask.max() > 0 :
        contours = cv2.findContours(bmask.astype(
                        np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        area = 0

        for cnt in contours:
            (x_, y_, w_, h_) = cv2.boundingRect(cnt)
            if w_*h_ > area:
                x = x_
                y = y_
                w = w_
                h = h_
                area = w_ * h_
        boxmask[y:y+h, x:x+w] = True
    return boxmask
