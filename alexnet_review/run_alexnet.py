#!/usr/bin/env python
# encoding: utf-8
'''
@author: wendell
@license: (C) Copyright 2018-2022, Node Supply Chain Manager Corporation Limited.
@contact: wendell@sina.com
@software: wendell
@file: .py
@time: 18-11-17 下午2:29
@desc:
'''

import os
import numpy as np
import tensorflow as tf
from alexnet.reload_alexnet.realexnet import Alexnet
import matplotlib.pyplot as plt

class_name = ['cat','dog']


def test_iamge(path_image,num_class,weights_path='Default'):
    img_string = tf.read_file(path_image)
    img_decoded = tf.image.decode_jpeg(img_string,channels=3)
    img_resized= tf.image.resize_images(img_decoded,[227,227])
    img_resized = tf.reshape(img_resized,shape=[1,227,227,3])

    model = Alexnet(img_resized,0.5,2,skip_layer='',weights_path=weights_path)
    score = tf.nn.softmax(model.fc8)
    max = tf.arg_max(score,1)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,'./temp.checkpints/model_epoch10.ckpt')

        print(sess.run(model.fc8))
        prob = sess.run(max)[0]

        plt.imshow(img_decoded.eval())
        plt.title('calss:',class_name[prob])
        plt.show()





if __name__=="__main__":
    test_iamge('./test/cat/cat.20.jpg','bvlc_alexnet.npy')

