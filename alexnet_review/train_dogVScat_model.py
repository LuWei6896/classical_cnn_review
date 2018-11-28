#!/usr/bin/env python
# encoding: utf-8
'''
@author: wendell
@license: (C) Copyright 2018-2022, Node Supply Chain Manager Corporation Limited.
@contact: wendell@sina.com
@software: wendell
@file: .py
@time: 18-11-9 上午9:40
@desc:
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import os
import cv2
import csv

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


# Display 10 images
DATA_DIR = "train"
for file_name in os.listdir(DATA_DIR)[0:10]:
    img_array = cv2.imread(os.path.join(DATA_DIR, file_name))
    plt.imshow(img_array)
    plt.show()
    print(file_name.split('.')[0])

#create data
DATA_DIR = "train"
IMG_SIZE = 28
def create_data():
    x = []
    y = []
    for file_name in os.listdir(DATA_DIR):
        img_array = cv2.imread(os.path.join(DATA_DIR, file_name))
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        x.append(new_array)
        if file_name.split('.')[0] == 'cat':
            y.append(0)
        else:
            y.append(1)
    return x, y

import random
x, y = create_data()
x = np.array(x)
y = np.array(y)
print(x.shape)
print(y.shape)

X_train = x / 255.0
Y_train = to_categorical(y, num_classes = 2)

print(X_train.shape)
print(Y_train.shape)

from keras import regularizers
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras import backend as K
import keras
from keras.models import load_model
import matplotlib.pyplot as plt

weight_decay = 1e-4
model = Sequential()
model.add(
    Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), input_shape=X_train.shape[1:]))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(2, activation='softmax'))
opt_rms = keras.optimizers.rmsprop(lr=0.001, decay=1e-6) #每次更新后学习速率的衰减量
model.compile(loss='categorical_crossentropy', optimizer=opt_rms, metrics=['accuracy'])


model.fit(x=X_train,
          y=Y_train, epochs=100, validation_split=0.1)

model.save_weights('dogVScat.h5')

model_json = model.to_json()
with open('model.json','w') as json_file:
    json_file.write(model_json)


model.save('dogVScat_CNN_weights.h5')

'''
#create data
DATA_DIR = "test1"
IMG_SIZE = 28
def create_test_data():
    x = []
    for i in range(1,12501):
        img_array = cv2.imread(os.path.join(DATA_DIR, str(i) + '.jpg'))
        new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        x.append(new_array)
    return x


# predict results
test = create_test_data()
test = np.array(test) / 255.0
results = model.predict(test)
print(results.shape)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="label")
submission = pd.concat([pd.Series(range(1,12500),name = "id"),results],axis = 1)

submission.to_csv("dogs_and_cats.csv",index=False)


'''