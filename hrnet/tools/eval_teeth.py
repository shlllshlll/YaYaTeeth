'''
@Author: shlll
@Date: 2020-09-07 14:36:02
@License: MIT License
@Email: shlll7347@gmail.com
@Modified by: shlll
@Last Modified time: 2020-09-07 14:36:04
@Description:
'''

import os
import argparse
from pathlib import Path

import torch
from tqdm.auto import tqdm
import cv2
import numpy as np
from PIL import Image, ImageDraw

from hrnet import models, datasets
from hrnet.datasets.base_dataset import BaseDataset
from hrnet.config import config
from hrnet.config import update_config
from hrnet.utils.utils import FullModel
from hrnet.core.criterion import CrossEntropy, OhemCrossEntropy


class HRNetModel(object):
    def __init__(self, model_path: str, cfg_path: str, model_type: str) -> None:
        args = type('args', (), {})()
        args.cfg = cfg_path
        args.opts = []
        update_config(config, args)

        self.base_dataset = BaseDataset()

        model = eval('models.'+config.MODEL.NAME + '.get_seg_model')(config)
        model = model.cuda()

        # load model
        print("=>Load model weight.")
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)

            if 'state_dict' in checkpoint.keys():
                for _ in range(len(checkpoint['state_dict'])):
                    k, v = checkpoint['state_dict'].popitem(False)
                    checkpoint['state_dict'][k[6:]
                                             if k.startswith('model.') else k] = v
                model.load_state_dict(checkpoint['state_dict'])
            else:
                for _ in range(len(checkpoint)):
                    k, v = checkpoint.popitem(False)
                    checkpoint[k[6:] if k.startswith('model.') else k] = v
                model.load_state_dict(checkpoint)
        else:
            raise FileNotFoundError("The model path do not exists.")

        # save loaded model
        self.model = model
        self.model_type = model_type

    def run(self, image: np.ndarray):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        resized_image = cv2.resize(
            image, tuple(config.TEST.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        resized_image = self.base_dataset.input_transform(resized_image)
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = np.expand_dims(resized_image, axis=0)
        image = torch.from_numpy(resized_image)
        image = image.cuda()
        seg_map = self.model(image)[0]
        if self.model_type == 'hrnetocr':
            seg_map = seg_map[0]
        seg_map = seg_map.detach()
        seg_map = torch.argmax(seg_map, dim=0)
        return resized_image, seg_map.cpu()

    def run_dir(self, image_dir: str, output_dir: str, save_vis: bool = False) -> None:
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(
                f"The input image directory '{str(image_dir)}' is not exists.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        vis_dir = image_dir / 'result_vis'
        if save_vis:
            vis_dir.mkdir(parents=True, exist_ok=True)

        for image_path in tqdm(list(image_dir.glob('*.jpg'))):
            print(image_path.name)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            _, seg_map = self.run(image)
            out_path = str(output_dir / (image_path.stem + '.npy'))
            np.save(out_path, seg_map)

            if save_vis:
                vis_path = vis_dir / (image_path.stem + '.png')
                vis_img = self.get_vis_result(image_path, seg_map)
                vis_img.save(vis_path)

    def get_vis_result(self, image_path, seg_map):
        # Open images and numpy arrays.
        src_img = Image.open(image_path).resize(seg_map.shape)

        # Draw result images.
        dst_img = src_img.copy()
        draw_img = ImageDraw.Draw(dst_img, mode='RGBA')
        x, y = np.where(seg_map == 2)
        dental_point = np.vstack((y, x)).T.flatten()
        draw_img.point(list(dental_point), fill=(255, 255, 0, 64))

        return dst_img


if __name__ == "__main__":
    pwd = Path(os.getcwd())
    graph_path = pwd / "output/hrnet/teeth0821/checkpoint.pth.tar"
    cfg_path = pwd / "config/hrnet.yaml"
    test_base_dir = pwd / "eval/20200829"
    image_dir = test_base_dir
    out_dir = test_base_dir / "outputs"
    print("=>Out dir: ", out_dir)

    model = HRNetModel(graph_path, cfg_path, 'hrnet')
    model.run_dir(image_dir, out_dir, save_vis=True)
