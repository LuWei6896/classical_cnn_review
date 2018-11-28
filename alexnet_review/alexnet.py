#!/usr/bin/env python
# encoding: utf-8
'''
@author: wendell
@license: (C) Copyright 2018-2022, Node Supply Chain Manager Corporation Limited.
@contact:
@software: wendell
@file: .py
@time: 18-11-15 上午9:45
@desc:
'''
import tensorflow as tf
import numpy as np

class Alexnet:
    def __init__(self,x,keepp_prob,num_classes,weights_path,skip_layer):
        self.keepp_prob = keepp_prob
        self.num_classses = num_classes
        self.weights_path = weights_path
        self.skip_layer = skip_layer
        self.x = x
        self.alexnet(x,keepp_prob,num_classes)

    def alexnet(self,x, keep_prob, num_classes):
        # conv1
        with tf.variable_scope('conv1') as scope:
            kernel = tf.get_variable('weights',shape=[11, 11, 3, 96])
            conv = tf.nn.conv2d(x, kernel, [1, 4, 4, 1], padding='SAME')
            biases = tf.get_variable('biases',shape=[96])
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name='conv1')

        # lrn1
        with tf.variable_scope('lrn1') as scope:
            lrn1 = tf.nn.local_response_normalization(conv1,
                                                      alpha=1e-4,
                                                      beta=0.75,
                                                      depth_radius=2,
                                                      bias=2.0)

        # pool1
        with tf.variable_scope('pool1') as scope:
            pool1 = tf.nn.max_pool(lrn1,
                                   ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='VALID')

        # conv2
        with tf.variable_scope('conv2') as scope:
            pool1_groups = tf.split(axis=3, value=pool1, num_or_size_splits=2)
            kernel = tf.get_variable('weights',shape=[5, 5, 48, 256])
            kernel_groups = tf.split(axis=3, value=kernel, num_or_size_splits=2)
            conv_up = tf.nn.conv2d(pool1_groups[0], kernel_groups[0], [1, 1, 1, 1], padding='SAME')
            conv_down = tf.nn.conv2d(pool1_groups[1], kernel_groups[1], [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases',shape=[256])
            biases_groups = tf.split(axis=0, value=biases, num_or_size_splits=2)
            bias_up = tf.nn.bias_add(conv_up, biases_groups[0])
            bias_down = tf.nn.bias_add(conv_down, biases_groups[1])
            bias = tf.concat(axis=3, values=[bias_up, bias_down])
            conv2 = tf.nn.relu(bias, name='conv2')

        # lrn2
        with tf.variable_scope('lrn2') as scope:
            lrn2 = tf.nn.local_response_normalization(conv2,
                                                      alpha=1e-4,
                                                      beta=0.75,
                                                      depth_radius=2,
                                                      bias=2.0)

        # pool2
        with tf.variable_scope('pool2') as scope:
            pool2 = tf.nn.max_pool(lrn2,
                                   ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='VALID')

            # conv3
        with tf.variable_scope('conv3') as scope:
            kernel = tf.get_variable('weights',shape=[3, 3, 256, 384])
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases',shape=[384])
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name='conv3')

        # conv4
        with tf.variable_scope('conv4') as scope:
            conv3_groups = tf.split(axis=3, value=conv3, num_or_size_splits=2)
            kernel = tf.get_variable('weights',shape=[3, 3, 192, 384])
            kernel_groups = tf.split(axis=3, value=kernel, num_or_size_splits=2)
            conv_up = tf.nn.conv2d(conv3_groups[0], kernel_groups[0], [1, 1, 1, 1], padding='SAME')
            conv_down = tf.nn.conv2d(conv3_groups[1], kernel_groups[1], [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases',shape=[384])
            biases_groups = tf.split(axis=0, value=biases, num_or_size_splits=2)
            bias_up = tf.nn.bias_add(conv_up, biases_groups[0])
            bias_down = tf.nn.bias_add(conv_down, biases_groups[1])
            bias = tf.concat(axis=3, values=[bias_up, bias_down])
            conv4 = tf.nn.relu(bias, name='conv4')

        # conv5
        with tf.variable_scope('conv5') as scope:
            conv4_groups = tf.split(axis=3, value=conv4, num_or_size_splits=2)
            kernel = tf.get_variable('weights',shape=[3, 3, 192, 256])
            kernel_groups = tf.split(axis=3, value=kernel, num_or_size_splits=2)
            conv_up = tf.nn.conv2d(conv4_groups[0], kernel_groups[0], [1, 1, 1, 1], padding='SAME')
            conv_down = tf.nn.conv2d(conv4_groups[1], kernel_groups[1], [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases',shape=[256])
            biases_groups = tf.split(axis=0, value=biases, num_or_size_splits=2)
            bias_up = tf.nn.bias_add(conv_up, biases_groups[0])
            bias_down = tf.nn.bias_add(conv_down, biases_groups[1])
            bias = tf.concat(axis=3, values=[bias_up, bias_down])
            conv5 = tf.nn.relu(bias, name='conv5')

        # pool5
        with tf.variable_scope('pool5') as scope:
            pool5 = tf.nn.max_pool(conv5,
                                   ksize=[1, 3, 3, 1],
                                   strides=[1, 2, 2, 1],
                                   padding='VALID', )

        # flattened6
        with tf.variable_scope('flattened6') as scope:
            flattened = tf.reshape(pool5, shape=[-1, 6 * 6 * 256])

        # fc6
        with tf.variable_scope('fc6') as scope:
            weights = tf.get_variable('weights',shape=[6 * 6 * 256, 4096])
            biases = tf.get_variable('biases',shape=[4096])
            bias = tf.nn.xw_plus_b(flattened, weights, biases)
            fc6 = tf.nn.relu(bias)

        # dropout6
        with tf.name_scope('dropout6') as scope:
            dropout6 = tf.nn.dropout(fc6, keep_prob)

        # fc7
        with tf.name_scope('fc7') as scope:
            weights = tf.Variable(tf.truncated_normal([4096, 4096],
                                                      dtype=tf.float32,
                                                      stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32),
                                 trainable=True, name='biases')
            bias = tf.nn.xw_plus_b(dropout6, weights, biases)
            fc7 = tf.nn.relu(bias)

        # dropout7
        with tf.name_scope('dropout7') as scope:
            dropout7 = tf.nn.dropout(fc7, keep_prob)

        # fc8
        with tf.name_scope('fc8') as scope:
            weights = tf.Variable(tf.truncated_normal([4096, num_classes],
                                                      dtype=tf.float32,
                                                      stddev=1e-1), name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=[num_classes], dtype=tf.float32),
                                 trainable=True, name='biases')
            fc8 = tf.nn.xw_plus_b(dropout7, weights, biases)
        self.output = fc8


    def load_initial_weights(self,session):
        '''load weights from file into network'''
        weights_dict = np.load(self.weights_path,encoding='bytes').item()

        var_list3 = [v for v in tf.trainable_variables()]
        #graph = tf.get_default_graph()

        for op_name in weights_dict:
            if op_name not in self.skip_layer:
                with tf.variable_scope(op_name,reuse=True):
                    for data in weights_dict[op_name]:
                        if len(data.shape) == 1:
                            var = tf.get_variable('biases',trainable=False)
                            session.run(var.assign(data))
                            #var1 = session.graph.get_tensor_by_name(op_name+'/'+'biases:0')
                            #test1 = tf.convert_to_tensor(data)
                            #session.run(tf.assign(session.graph.get_tensor_by_name(op_name+'/'+'biases:0'),tf.convert_to_tensor(data)))

                        else:
                            var1 = tf.get_variable('weights',trainable=False)
                            session.run(var1.assign(data))
                            #var2 = session.graph.get_tensor_by_name(op_name+'/'+'weights:0')
                            #test2 = tf.convert_to_tensor(data)
                            #session.run(tf.assign(session.graph.get_tensor_by_name(op_name+'/'+'weights:0'),tf.convert_to_tensor(data)))
                            #print(session.run(test2))
                            #print('1')

        var_list4 = [v for v in tf.trainable_variables() if v.name.split('/')[0] not in self.skip_layer]
        for show in var_list4:
            if len(show.shape)==1:
                print('biases',show.name,session.run(show))
            else:
                print('weights',show.name,session.run(show))
            print('load value')
        print('1')



    def conv(self,x,filter_height,filter_width,filter_num,stride_y,stride_x,name,padding='SAME',groups=1):
        input_channel = int(x.get_shape()[-1])

        convole = lambda i,k:tf.nn.conv2d(i,k,strides=[1,stride_x,stride_y,1],padding=padding)

        with tf.variable_scope(name) as scope:
            weights = tf.get_variable('weights',shape=[filter_width,filter_height,np.floor(input_channel/groups),filter_num])
            biases = tf.get_variable('biases',shape=[filter_num])

        if groups == 1:
            conv =convole(x,weights)
        else:
            input_grops = tf.split(aixs =3 ,num_or_size_splits=groups,value=x)
            weight_grops = tf.split(aix=3,num_or_size_splits=groups,value=weights)
            #biases = tf.split(aixs=0,num_or_size_splits=groups,value=biases)
            output_groups = [convole(i,k) for i,k in zip(input_grops,weight_grops)]
            conv = tf.concat(aixs=3,values=output_groups)

        bias = tf.reshape(tf.nn.bias_add(conv,biases),tf.shape(conv))
        relu = tf.nn.relu(bias,name=scope.name)

        return relu








