from torchvision import transforms as T
import torch
from argparse import ArgumentParser
import einops
from ..utils.KeyTypes import extract_ascending_list
from ipdb import set_trace

class TrAugmentImage() :
    """
    Data augmentation techniques for images
    Augments all 'Image' fields in the sequence ( with )

    Args :
        Name image_augmentation (list str) : data augmentation to return
    """
    def __init__(self, image_augmentation) :
        self.augs = []
        augs_names =  image_augmentation.split('_')
        for name in augs_names :
            self.interpret_name(name)
        self.augs.append(self.normalisation())
        self.declare()

    def interpret_name(self, name) :
        if (name == 'none') or (name=='') :
            pass
        elif 'randillumination':
            aug = T.RandomChoice(transforms=[T.Grayscale(3),
                                            T.ColorJitter(0.5, 0.3, 0.3),
                                            T.GaussianBlur(5),
                                            T.RandomInvert(1),
                                            T.RandomAdjustSharpness(2),
                                            T.RandomAutocontrast()])
            self.augs.append(aug)
        else :
            raise Exception(f'Image augmentation {name} is unknown')

    @staticmethod
    def normalisation() :
        return T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


    def __call__(self, ret) :
        """
        Call all augmentations defined in the init
        ret : dict with keys 'Image' ( 3 I J)
         1. We assemble all ret into a flow volumet ( 3 T I J) so we can treat them all at once
         2. Apply augmentation on the flow volume
         3. Split the volume into chunks and redistribute the flow
         BEWARE : in this function we use a "default" order for the time given in the key dict
         above. So the volume will be ordered following this. This is important for cases where
         the flow augmentation vary with time.
        """
        if len(self.augs) > 0 :
            # Get all keys both in ret and with type flow in the order defined in KEY_TYPES
            image_keys = extract_ascending_list(ret.keys(), 'Image')
            if len(image_keys) > 0 :
                image_volume = einops.rearrange([ret[k] for k in image_keys], 't c i j -> t c i j', c=3)

                # Apply augmentations to the flow volume
                for aug in self.augs :
                    image_volume = aug(image_volume)

                # Redistribute flow to their respective spot in the dict
                for i in range(len(image_keys)) :
                     ret[image_keys[i]] = image_volume[i]
        return ret

    def declare(self):
        print(f'Image Transformations : {[aug for aug in self.augs]}')


    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--image_augmentation', type=str, default='none')
        return parser
