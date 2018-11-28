#!/usr/bin/env python
# encoding: utf-8
'''
@author: wendell
@license: (C) Copyright 2018-2022, Node Supply Chain Manager Corporation Limited.
@contact:
@software: wendell
@file: .py
@time: 18-11-15 下午2:20
@desc:
'''

import os
import numpy as np
import tensorflow as tf
from dogVScat.alexnet import Alexnet
from dogVScat.alexnet_recode import AlexNet
from dogVScat.img_generator import ImageDataGenerator
from datetime import datetime
import glob

def main():
    learning_rate = 1e-2
    num_epochs = 10
    train_batch_size = 128
    test_batch_size = 128
    dropout_rate = 0.5
    num_classes = 2
    display_step = 10

    filewriter_path = './tmp/tensorboard'
    checkpoint_path = './tmp/checkpoints'
    weights_path = 'bvlc_alexnet.npy'

    skip_layer = ['fc8','fc7']

    image_format = 'jpg'
    file_name_of_class = ['cat','dog']
    train_dataset_paths = ['/home/luwei/python_project2/dogVScat/train/cat','/home/luwei/python_project2/dogVScat/train/dog']
    test_dataset_paths = ['/home/luwei/python_project2/dogVScat/test/cat/','/home/luwei/python_project2/dogVScat/test/dog']

    train_image_paths = []
    train_labels = []

    for train_dataset_path in train_dataset_paths:
        length = len(train_image_paths)
        train_image_paths[length:length] = np.array(glob.glob(train_dataset_path + '*.'+ image_format)).tolist()
    for image_path in train_image_paths:
        image_file_name = image_path.split('/')[-1]
        for i in range(num_classes):
            if file_name_of_class[i] in image_file_name:
                train_labels.append(i)
                break

    test_image_paths = []
    test_labels =[]

    for test_dataset_path in test_dataset_paths:
        length = len(test_image_paths)
        test_image_paths[length:length] = np.array(glob.glob(test_dataset_path + '*.' + image_format)).tolist()
    for image_path in test_image_paths:
        image_file_name = image_path.split('/')[-1]
        for i in range(num_classes):
            if file_name_of_class[i] in image_file_name:
                test_labels.append(i)
                break

    train_data = ImageDataGenerator(
        images = train_image_paths,
        labels =train_labels,
        batch_size = train_batch_size,
        num_classes = num_classes,
        image_format = image_format,
        shuffle = True)

    test_data = ImageDataGenerator(
        images = test_image_paths,
        labels = test_labels,
        batch_size = test_batch_size,
        num_classes = num_classes,
        image_format = image_format,
        shuffle = False)

    with tf.name_scope('input'):
        train_iterator = tf.data.Iterator.from_structure(train_data.data.output_types,
                                                         train_data.data.output_shapes)

        training_initalizer = train_iterator.make_initializer(train_data.data)
        test_iterator = tf.data.Iterator.from_structure(test_data.data.output_types,
                                                        test_data.data.output_shapes)
        test_initalizer = test_iterator.make_initializer(test_data.data)

        train_next_batch = train_iterator.get_next()
        test_next_batch = test_iterator.get_next()


    x = tf.placeholder(tf.float32,[None,227,227,3])
    y = tf.placeholder(tf.float32,[None,num_classes])
    keep_prob = tf.placeholder(tf.float32)

    model = Alexnet(x,keep_prob,num_classes=num_classes,weights_path=weights_path,skip_layer=skip_layer)

    score = model.output

    var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in skip_layer]
    #var_list_test = [v for v in tf.trainable_variables()]

    with tf.name_scope('loss'):
        loos_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,labels=y))

    with tf.name_scope('optimizer'):
        gradients = tf.gradients(loos_op,var_list)
        gradients = list(zip(gradients,var_list))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.apply_gradients(grads_and_vars=gradients)

    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(score,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    init = tf.global_variables_initializer()

    tf.summary.scalar('loss',loos_op)
    tf.summary.scalar('accuracy',accuracy)


    for gradient,var in gradients:
        if var.name.split('/')[1] =='weights:0':
            tf.summary.histogram(var.name+'/gradient',gradient)


    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(filewriter_path)


    saver = tf.train.Saver()

    train_batches_per_epoch = int(np.floor(train_data.data_size / train_batch_size))
    test_batches_per_epoch = int(np.floor(test_data.data_size / test_batch_size))

    with tf.Session() as sess:
        sess.run(init)
        writer.add_graph(sess.graph)
        model.load_initial_weights(sess)

        var_list2 = [v for v in tf.trainable_variables() if v.name.split('/')[0] not in skip_layer]

        print('{}: Start training ...'.format(datetime.now()))
        print('{}: Open Tensorboard at --logdir {}'.format(datetime.now(),
                                                           filewriter_path))
        for epoch in range(num_epochs):
            sess.run(training_initalizer)
            print('{}: Epoch number: {} start'.format(datetime.now(),epoch+1))

            for step in range(train_batches_per_epoch):
                img_batch,label_batch = sess.run(train_next_batch)
                loss,_ = sess.run([loos_op,train_op],feed_dict={
                    x:img_batch,
                    y:label_batch,
                    keep_prob:dropout_rate})
                if step % display_step == 0:
                    print('{} : loss = {}'.format(datetime.now(),loss))

                    s = sess.run(merged_summary,feed_dict={
                        x:img_batch,
                        y:label_batch,
                        keep_prob:1.
                    })

                    writer.add_summary(s,epoch * train_batches_per_epoch + step)

            print('{}: start validation'.format(datetime.now()))
            sess.run(test_initalizer)
            test_acc =0.
            test_count =0

            for _ in range(test_batches_per_epoch):
                img_batch,label_batch = sess.run(test_next_batch)
                acc = sess.run(accuracy,feed_dict={
                    x:img_batch,
                    y:label_batch,
                    keep_prob:1.0
                })
                test_acc += acc
                test_count +=1
            try:
                test_acc /= test_count
            except:
                print('ZeroDivisionError!')
            print("{}: Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

            # save model
            print("{}: Saving checkpoint of model...".format(datetime.now()))
            checkpoint_name = os.path.join(checkpoint_path,'model_epoch' + str(epoch + 1)+ '.ckpt')
            save_path = saver.save(sess,checkpoint_name)

            print('{}: epoch number:{} end'.format(datetime.now(),epoch +1))




if __name__ == '__main__':
    main()












