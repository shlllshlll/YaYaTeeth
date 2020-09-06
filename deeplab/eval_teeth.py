'''
@Author: shlll
@Date: 2020-04-17 11:00:31
@License: MIT License
@Email: shlll7347@gmail.com
@Modified by: shlll
@Last Modified time: 2020-04-17 11:21:17
@Description:
'''

import os
import argparse
from pathlib import Path

import tensorflow as tf
from PIL import Image
import numpy as np
from tqdm.auto import tqdm


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def __init__(self, graph_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()
        graph_path = Path(graph_path)
        if not graph_path.exists():
            raise FileNotFoundError("The graph path not exists.")

        graph_def = tf.compat.v1.GraphDef()
        loaded = graph_def.ParseFromString(open(graph_path, 'rb').read())

        if graph_def is None:
            raise RuntimeError('Graph load failed.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        config = tf.ConfigProto()
        # config.gpu_options.visible_device_list = '1'
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.graph, config=config)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(
            target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

    def run_dir(self, image_dir, output_dir):
        image_dir = Path(image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(
                f"The input image directory '{str(image_dir)}' is not exists.")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for image_path in tqdm(list(image_dir.glob('*'))):
            image = Image.open(image_path)
            _, seg_map = self.run(image)
            out_path = str(output_dir / (image_path.stem + '.npy'))
            np.save(out_path, seg_map)


if __name__ == "__main__":
    paser = argparse.ArgumentParser(description="Teeth result evaluation.")
    paser.add_argument("-o", "--out_dir", type=str, default="result",
                       help="The relative output directory.")
    args = paser.parse_args()

    pwd = Path(os.getcwd())
    graph_path = pwd / "trained_model/frozen_inference_graph.pb"
    test_base_dir = pwd / "eval/full-VOC_adult_20200512"
    image_dir = test_base_dir / "JPEGImages"
    out_dir = test_base_dir / args.out_dir
    print("=>Out dir: ", out_dir)

    model = DeepLabModel(graph_path)
    model.run_dir(image_dir, out_dir)
