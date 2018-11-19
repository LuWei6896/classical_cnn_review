#!/usr/bin/env python
# encoding: utf-8
'''
@author: wendell
@license: (C) Copyright 2018-2022, Node Supply Chain Manager Corporation Limited.
@contact:
@software: wendell
@file: .py
@time: 18-11-6 上午11:40
@desc:
'''

from __future__  import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from six.moves import xrange
import tensorflow as tf
import os

NUM_CLASS = 2
IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_CHANNELS = 3

def read_image(root_path,batch_size):
    image_paths =[]
    labels = []
    label =0
    classes = sorted(os.walk(root_path).__next__()[1])
    for c in classes:
        c_dir = os.path.join(root_path,c)
        walk = os.walk(c_dir).__next__()[2]
        for sample in walk:
            if sample.endswith('.jpg') and sample.endswith('.jpeg'):
                image_paths.append(os.path.join(c_dir,sample))
                labels.append(label)
        label +=1

    image_paths = tf.convert_to_tensor(image_paths,tf.string)
    labels = tf.convert_to_tensor(labels,tf.int32)

    image_path, label = tf.train.slice_input_producer([image_paths,labels],shuffle=True)

    image_tensor = tf.read_file(image_path)
    image_decode = tf.image.decode_jpeg(image_tensor,channels=IMAGE_CHANNELS)

    image_resize = tf.image.resize_images(image_decode,size=(IMAGE_HEIGHT,IMAGE_WIDTH))
    image_regularization = image_resize*1.0/127.5 -1.0

    X,Y =tf.train.batch([image_regularization,label],batch_size=batch_size,num_threads=4,capacity=batch_size*8)

