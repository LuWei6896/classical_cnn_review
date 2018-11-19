#!/usr/bin/env python
# encoding: utf-8
'''
@author: wendell
@license: (C) Copyright 2018-2022, Node Supply Chain Manager Corporation Limited.
@contact: wendell@sina.com
@software: wendell
@file: .py
@time: 18-11-19 下午3:45
@desc:
'''
import numpy as np
import tensorflow as tf

def load_initial_weights(self, session):
    # Load the weights into memory
    weights_dict = np.load(self.WEIGHTS_PATH, encoding = 'bytes').item()

    # Loop over all layer names stored in the weights dict
    for op_name in weights_dict:

        # Check if the layer is one of the layers that should be reinitialized
        if op_name not in self.SKIP_LAYER:
            with tf.variable_scope(op_name, reuse = True):
                # Loop over list of weights/biases and assign them to their corresponding tf variable
                for data in weights_dict[op_name]:
                    # Biases
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases', trainable = False)
                        session.run(var.assign(data))

                    # Weights
                    else:
                        var = tf.get_variable('weights', trainable = False)
                        session.run(var.assign(data))


if __name__=='__main__':
    with tf.Session() as sess:
        load_initial_weights()