#!/usr/bin/env python
# encoding: utf-8
'''
@author: wendell
@license: (C) Copyright 2018-2022, Node Supply Chain Manager Corporation Limited.
@contact:
@software: wendell
@file: .py
@time: 18-11-12 上午9:13
@desc:
'''

import pandas as pd
import numpy as np

import os
import cv2

from keras.models import Sequential

import keras
from keras.models import model_from_json
from keras.models import load_model
import shutil
import matplotlib.pyplot as plt

model = Sequential() #序惯模型

#加载cnn net
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

#load weights
loaded_model.load_weights('dogVScat.h5')


#load cnn-net & weight
model_ = load_model('dogVScat_CNN_weights.h5')



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
results = model_.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)


results = pd.Series(results,name="label")

submission = pd.concat([pd.Series(range(1,12500),name = "id"),results],axis = 1)

cat_ = submission[(submission.label==0)].index.tolist()


img_path_list = list(map(lambda index:os.path.join('test1',str(index+1)+'.jpg'),cat_))


ok = map(lambda img:shutil.move(img,'cat'),img_path_list)
ok1 = list(ok)
print('1')
#cat_dataframe = submission[(submission.label==0)]
#cat_dataframe.to_csv("test.csv",index=False)