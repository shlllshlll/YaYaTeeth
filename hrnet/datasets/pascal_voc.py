'''
@Author: shlll
@Date: 2020-08-28 1:23:43
@License: MIT License
@Email: shlll7347@gmail.com
@Modified by: shlll
@Last Modified time: 2020-08-28 1:24:25
@Description:
'''

import os
from pathlib import Path

import cv2
import numpy as np

import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset
from ..config import config

class PascalVOC(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=20,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=700,
                 crop_size=(512, 512),
                 downsample_rate=1,
                 scale_factor=11,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        super(PascalVOC, self).__init__(ignore_label, base_size,
                crop_size, downsample_rate, scale_factor, mean, std)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None

        self.multi_scale = multi_scale
        self.flip = flip

        self.img_dir = Path(self.root) / 'JPEGImages'
        self.gt_dir = Path(self.root) / 'SegmentationClass_filted'
        self.anno_dir = Path(self.root) / 'ImageSets/Segmentation'
        self.img_list = [line.strip().split() for line in open(self.anno_dir / list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]

    def read_files(self):
        files = []
        for name in self.img_list:
            name = name[0]
            image_path = str(self.img_dir / (name + '.jpg'))
            label_path = str(self.gt_dir / (name + '.png'))

            sample = {"img": image_path,
                      "label": label_path,
                      "name": name,}
            files.append(sample)
        return files

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]

        image = cv2.imread(item['img'], cv2.IMREAD_COLOR)
        label = cv2.imread(item['label'], cv2.IMREAD_GRAYSCALE)
        size = label.shape

        if 'testval' in self.list_path:
            image = cv2.resize(image, self.crop_size,
                               interpolation = cv2.INTER_LINEAR)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        image, label = self.resize_image(image, label, self.crop_size)
        image, label = self.gen_sample(image, label,
                                self.multi_scale, False)

        return image.copy(), label.copy(), np.array(size), name

    def inference(self, model, image, flip):
        size = image.size()
        pred = model(image)
        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )

        return pred.exp()
