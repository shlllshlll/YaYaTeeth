'''
@Author: shlll
@Date: 2020-09-07 11:19:36
@License: MIT License
@Email: shlll7347@gmail.com
@Modified by: shlll
@Last Modified time: 2020-09-07 11:20:34
@Description:
'''

from pathlib import Path

from tqdm.auto import tqdm
import cv2

from hrnet.config import config

def filter_dataset():
    base_dir = Path(config.DATASET.ROOT)
    if not base_dir.exists():
        raise Exception("The dataset dir not exists.")

    gt_dir = base_dir / 'SegmentationClass'
    filted_gt_dir = base_dir / 'SegmentationClass_filted'

    if filted_gt_dir.exists():
        return
    filted_gt_dir.mkdir(exist_ok=True)

    print('=> Start filtering datasets')
    for img_path in tqdm(gt_dir.glob('*')):
        im = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        im[im>=3]=0
        cv2.imwrite(str(filted_gt_dir / img_path.name), im)
