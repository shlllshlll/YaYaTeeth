'''
@Author: shlll
@Date: 2020-04-17 15:58:00
@License: MIT License
@Email: shlll7347@gmail.com
@Modified by: shlll
@Last Modified time: 2020-04-17 15:58:04
@Description:
'''

import tensorflow as tf
from tensorflow.python.client import device_lib


def set_gpu_growth(gpu_num=0):
    device_lib.list_local_devices()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        if len(gpus) > gpu_num:
            gpu = gpus[gpu_num]
        else:
            gpu = gpus[0]

        try:
            tf.config.experimental.set_visible_devices(gpu, 'GPU')
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices(
                'GPU')
            print(len(gpus), "Physical GPUs,",
                  len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


def force_gpu_growth():
    oldinit = tf.Session.__init__
    def myinit(session_object, target='', graph=None, config=None):
        print("Intercepted!")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
    tf.Session.__init__ = myinit
