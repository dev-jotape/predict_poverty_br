# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 23:59:29 2019

@author: diego
"""

### Libraries --------------------------------------------------------------

# import plaidml.keras # Required if using AMD GPU
# plaidml.keras.install_backend() # Required if using AMD GPU
import keras
from keras.applications.vgg16 import VGG16
import tensorflow.keras.utils as utils
from keras.models import Sequential, load_model
from keras.applications.imagenet_utils import preprocess_input
from keras.layers.convolutional import Conv2D, AveragePooling2D
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from tensorflow.keras.optimizers import SGD
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers import Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import json as simplejson
import os
import gc


### get image featuers -----------------------------------------------------

# model_old = VGG16(weights='imagenet', include_top=False)

### Append correct label to data -------------------------------------------

# all_figures = []
# trainLabels = []

# path_1 = '../../dataset/google_images/class1/'
# class_1_files = os.listdir(path_1)
# trainLabels += [0] * len(class_1_files)
# all_figures += [path_1 + i for i in class_1_files]

# path_2 = '../../dataset/google_images/class2/'
# class_2_files = os.listdir(path_2)
# trainLabels += [1] * len(class_2_files)
# all_figures += [path_2 + i for i in class_2_files]

# path_3 = '../../dataset/google_images/class3/'
# class_3_files = os.listdir(path_3)
# trainLabels += [2] * len(class_3_files)
# all_figures += [path_3 + i for i in class_3_files]

# trainData = []

# print('GET IMAGE FEATURES ----------')

# def get_input_feature(img_path):
#     try:
#         img = utils.load_img(img_path)
#         x = utils.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#         features = model_old.predict(x)
#         return features[0]
#     except NameError:
#         print('error => ', img_path)
#         print('error => ', NameError)
#         return None

# trainData = []
# for idx, i in enumerate(all_figures):
#     print(idx)
#     a = get_input_feature(i)
#     if a is not None:
#         trainData.append(a)


# x_all = np.asarray(trainData)
# y_all = np.asarray(trainLabels)

### Save dataset --------------------------------------------------------------
# np.save('../../dataset/x_all.npy', x_all)
# np.save('../../dataset/y_all.npy', y_all)


# del(a, all_figures, class_1_files, class_2_files, class_3_files, i, idx, 
#     path_1, path_2, path_3, trainData, trainLabels)
# gc.collect()

### Load dataset --------------------------------------------------------------

x_all = np.load('../../dataset/x_all.npy')
y_all = np.load('../../dataset/y_all.npy')

y_all = utils.to_categorical(y_all, num_classes=3)

### Split data into training and testing -----------------------------------

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Divida o conjunto de dados em conjunto de treinamento e teste
x_train_val, x_test, y_train_val, y_test = train_test_split(x_all, y_all, stratify=y_all, test_size=test_ratio, random_state=123)

# Divida o conjunto de treinamento em conjunto de treinamento e validação
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, stratify=y_train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=123)


del(x_all, y_all)
gc.collect()

### Define CNN model configuration -----------------------------------------

# model = Sequential()
# model.add(Conv2D(4096, (6, 6), 
#                  activation='relu', 
#                  input_shape=(12, 12, 512), 
#                  strides=(6, 6), 
#                  name='input'))
# model.add(Dropout(0.5))
# model.add(Conv2D(4096, (1, 1), 
#                  activation='relu', 
#                  strides=(1, 1), 
#                  name='conv_7'))
# model.add(Dropout(0.5))
# model.add(Conv2D(4096, (1, 1), 
#                  strides=(1, 1), 
#                  name='conv_8'))
# model.add(AveragePooling2D((2, 2), 
#                            strides=(1, 1), 
#                            name='add_pool'))

# model.add(Flatten(name='flatten'))
# model.add(Dense(3))
# model.add(Activation("softmax"))

# opt = SGD(learning_rate=1e-2)

# ### Compile model ----------------------------------------------------------

# METRICS = [
#       "accuracy",
#       keras.metrics.TruePositives(name='tp'),
#       keras.metrics.FalsePositives(name='fp'),
#       keras.metrics.TrueNegatives(name='tn'),
#       keras.metrics.FalseNegatives(name='fn'), 
#       keras.metrics.Precision(name='precision'),
#       keras.metrics.Recall(name='recall'),
#       keras.metrics.AUC(name='auc'),
#       keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
# ]

# model.compile(loss="categorical_crossentropy", 
#               optimizer=opt, 
#               metrics=METRICS)

# class AccuracyHistory(Callback):
#     def on_train_begin(self, logs={}):
#         self.acc = []

#     def on_epoch_end(self, batch, logs={}):
#         self.acc.append(logs.get('acc'))

# history = AccuracyHistory()


# ### Fit model (storing  weights) -------------------------------------------

# filepath="./weights/weights_baseline.hdf5"
# checkpoint = ModelCheckpoint(filepath, 
#                              monitor='val_accuracy', 
#                              verbose=1, 
#                              save_best_only=True, 
#                              mode='max')

# # Add early stopping
# es = EarlyStopping(monitor='val_accuracy', verbose=1, patience=10)

# callbacks_list = [checkpoint, history, es]

# model.fit(x_train, y_train, 
#           validation_data=(x_val, y_val),
#           batch_size=32, 
#           epochs=100, 
#           verbose=1,
#           callbacks=callbacks_list)

# =============================================================================
model = load_model('./weights/weights_baseline.hdf5')
# 
# model.fit(x_train, y_train, 
#           validation_split=0.2,
#           batch_size=32, 
#           epochs=15, 
#           verbose=1,
#           callbacks=callbacks_list)
# =============================================================================


### storing Model in JSON --------------------------------------------------

# model_json = model.to_json()

# with open("./model/model_baseline.json", "w") as json_file:
#     json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))


### evaluate model ---------------------------------------------------------

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 

