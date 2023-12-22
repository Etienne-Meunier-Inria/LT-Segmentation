# In this script we implement the Biou evaluation of the paper :
# Divided Attention: Unsupervised Multi-Object Discovery with Contextually Separated Slots
# The idea is that each gt is associated to it's most likely segment
# "We match each ground truth object mask to the most likely segment (with the highest IoU),
# and then compute bIoU by averaging this IoU across all objects in the dataset."
#
# For x in gt_annotations_including_background
#   maxiou(x) = 0
#   For y i segments_predicted :
#       maxiou(x) = max(maxiou(x), iou(x, y))
# Piou = mean(maxiou(x))

import sys; from __init__ import PRP; sys.path.append(PRP)

from PIL import Image
from evaluations.utils_evals import metric_grid
from utils.evaluations import db_eval_iou

base_gt = "PATH TO GT FOLDER"#'/net/serpico-fs2/emeunier/FBMS/FBMS_diva/Gt_diva/Testset/'

cs = None
base_pred = "PATH TO PRED FOLDER" # f'SavedModels/2r9flv5x_copy/FBMS/FBMS_clean/Results_cs={cs}/'


gts = glob(base_gt+'*/*.png')

scores = []
for gt_path in gts :
    pred_path = gt_path.replace(base_gt, base_pred)
    gt = torch.tensor(np.array(Image.open(gt_path)))[None]
    pred = torch.tensor(np.array(Image.open(pred_path)))[None]

    metric, labels_pred, labels_gt = metric_grid(pred, gt, db_eval_iou, exclude_minus_1_gt=False)
    scs = list(metric.max(axis=1).values[0].numpy())
    for i, s in enumerate(scs) :
        scores.append((gt_path, f'object_{i}', s))


df = pd.DataFrame(scores, columns=['Path', 'Object', 'Score'])

df['Sequence'] = df['Path'].apply(lambda x : x.split('/')[-2])


# We compute the score for all instances in each sequence and then average over all sequences
dfgb = df.groupby(['Sequence']).mean()
dfgb
dfgb.mean()
