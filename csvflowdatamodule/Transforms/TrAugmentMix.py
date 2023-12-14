from torchvision import transforms as T
import torch
from argparse import ArgumentParser
import einops
from ..utils.KeyTypes import extract_ascending_list
from ipdb import set_trace

def resizecrop(tensor) :
    """
    tensor (*, H, W)
    return resized crop tensor
    """
    H, W = tensor.shape[-2:]
    return T.RandomResizedCrop(size=(H,W))(tensor)

class TrAugmentMix() :
    """
    Data augmentation techniques for images
    Augments all 'Image' fields in the sequence ( with )

    Args :
        Name image_augmentation (list str) : data augmentation to return
    """
    def __init__(self, mix_augmentation) :
        self.augs = []
        augs_names =  mix_augmentation.split('_')
        for name in augs_names :
            self.interpret_name(name)
        self.declare()

    def interpret_name(self, name) :
        if (name == 'none') or (name=='') :
            pass
        elif 'randspatial':
            aug = T.RandomChoice(transforms=[T.RandomVerticalFlip(0.9),
                                             T.RandomHorizontalFlip(0.9),
                                             T.ElasticTransform(alpha=25.),
                                             T.Lambda(resizecrop)])
            self.augs.append(aug)
        else :
            raise Exception(f'Mix augmentation {name} is unknown')



    def __call__(self, ret) :
        """
        Call all augmentations defined in the init
        ret : dict with keys 'Image' ( 3 I J) and 'Flows' ( 2 I J)
         1. We assemble all ret into a volume ( C*T I J) so we can treat them all at once
         2. Apply augmentation on the volume
         3. Split the volume into chunks and redistribute the data types
         BEWARE : in this function we use a "default" order for the time given in the key dict
         above. So the volume will be ordered following this. This is important for cases where
         the flow augmentation vary with time.
        """
        if len(self.augs) > 0 :
            image_keys = extract_ascending_list(ret.keys(), 'Image')
            flow_keys = extract_ascending_list(ret.keys(), 'Flow')
            mask_keys = extract_ascending_list(ret.keys(), 'GtMask')
            list_keys = image_keys + flow_keys + mask_keys
            if len(list_keys) > 0 :
                list_mat = [ret[k] for k in list_keys]
                packed, ps = einops.pack(list_mat, '* i j')
                for aug in self.augs :
                    packed = aug(packed)
                list_mat = einops.unpack(packed, ps, '* i j')
                for i, k in enumerate(list_keys) :
                    ret[k] = list_mat[i]
        return ret

    def declare(self):
        print(f'Mix Transformations : {[aug for aug in self.augs]}')


    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--mix_augmentation', type=str, default='none')
        return parser
