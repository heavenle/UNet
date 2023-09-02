import os
import cv2
import numpy as np
import torch
from utils.Registry import LOADER


def mask2onehotmask(mask, config):
    one_hot_mask = []
    for class_value in config['class_value']:
        one_hot_mask.append(np.where(mask == int(class_value), 1, 0).astype(np.float32))
    one_hot_mask = np.dstack(one_hot_mask)
    return one_hot_mask


@LOADER.registry()
class NormalDataLoader(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, opt, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            opt(dict): parameter
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.jpg
            │   ├── 0aab0a.jpg
            │   ├── 0b1761.jpg
            │   ├── ...
            |
            └── masks
                ├── 0a7e06.png
                ├── 0aab0a.png
                ├── 0b1761.png
                ├── ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform
        self.config = opt

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]

        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))
        img = cv2.resize(img, (self.config['resize_w'], self.config['resize_h']))
        img = img.astype(np.uint8) / 255.
        img = img.transpose(2, 0, 1)

        mask = cv2.imread(os.path.join(self.mask_dir, img_id + self.mask_ext), 0)
        mask = cv2.resize(mask, (self.config['resize_w'], self.config['resize_h']))
        # mask = mask[np.newaxis, :, :]
        return {"img": img,
                "mask": mask}
