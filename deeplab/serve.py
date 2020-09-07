# /data2/shuting/shuting/master_service_pzn_20200318.py
# /data2/pzn2/server_test20200318

from pathlib import Path
import shutil
import tensorflow as tf
import numpy as np
import cv2 as cv
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from matplotlib import pyplot as plt
import matplotlib as mpl
import time
import os
from matplotlib.font_manager import FontProperties
from PIL import Image

# create Session
graph = tf.Graph()
INPUT_TENSOR_NAME = 'ImageTensor:0'
OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
graph_def = None
# model path
# graph_path = "./models/frozen_inference_graph_0705_700_700_726.pb" #pang_fix_20190705
# graph_path = "./models/frozen_inference_graph_20200315_700_700_708.pb" #pang_fix_20190705
graph_path = "./trained_model/frozen_inference_graph.pb"  # pang_fix_20190705

graph_def = tf.compat.v1.GraphDef()
loaded = graph_def.ParseFromString(open(graph_path, 'rb').read())

if graph_def is None:
    raise RuntimeError('Graph load failed.')

with graph.as_default():
    tf.import_graph_def(graph_def, name='')

config = tf.ConfigProto()
# config.gpu_options.visible_device_list = '1'
config.gpu_options.allow_growth = True

sess = tf.Session(graph=graph, config=config)

# plt_base_path = Path('plt_result')
# res_base_path = Path('result')

# if plt_base_path.exists():
#     shutil.rmtree(plt_base_path)
# if res_base_path.exists():
#     shutil.rmtree(res_base_path)
# plt_base_path.mkdir()
# res_base_path.mkdir()

# start test
# rootdir = os.path.abspath('JPEGImages/')#image folder
rootdir = os.path.abspath('eval/full-VOC(adult)20200510/JPEGImages/')  # image folder
dstpath = Path('eval/full-VOC(adult)20200510/result_img/')
dstpath.mkdir(exist_ok=True)

imglist = os.listdir(rootdir)
timesum = 0
for i in range(0, len(imglist)):
    print('The image name is:', imglist[i])
    starttime = time.time()
    print('The starttime is:', starttime)

    imgpath = os.path.join(rootdir, imglist[i])

    img = load_img(imgpath).resize((513, 513))  # get image
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0).astype(np.uint8)  # uint8是之前导出模型时定义的
    # use models
    result = sess.run(
        OUTPUT_TENSOR_NAME,
        feed_dict={INPUT_TENSOR_NAME: img})

    # trans result
    # Save numpy result
    # np.save(str(res_base_path / imglist[i].split('.')[0]), result)

    # save the (0-1-2) result
    result_trans = result.transpose((1, 2, 0))
    result_yuan_filename = 'eval/full-VOC(adult)20200510/result_img/' + \
        imglist[i]  # (0-1-2) result path
    result_trans = cv.resize(result_trans, (700, 700))
    cv.imwrite(result_yuan_filename, result_trans)

