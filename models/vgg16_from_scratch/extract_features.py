import tensorflow as tf
import keras.utils as utils
from keras.applications.imagenet_utils import preprocess_input
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Dropout, Flatten, Dense, Activation
import numpy as np
from keras.optimizers import Adam
import json as simplejson
import pandas as pd
from keras.models import load_model
from keras.layers.convolutional import Conv2D, AveragePooling2D

# import os
# import time

print('IMPORT DATA -----------------------------')
input_shape = (224,224,3)

# def process_input(img_path):
#     try:
#         img = utils.load_img(img_path, target_size=input_shape)
#         x = utils.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#         return x
#     except NameError:
#         print('error => ', img_path)
#         print('error => ', NameError)
#         return None


# ### Append correct label to data -------------------------------------------

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

# trainData = []
# t1 = time.time()
# for idx, i in enumerate(all_figures):
#     # print(idx)
#     a = process_input(i)
#     if a is not None:
#         trainData.append(a)


# x_all = np.asarray(trainData)
# y_all = np.asarray(trainLabels)

### Save data --------------------------------------------------------------
# np.save('../../dataset/x_all_v2.npy', x_all)
# np.save('../../dataset/y_all_v2.npy', y_all)

# x_all = np.load('../../dataset/x_all_v2.npy')
# y_all = np.load('../../dataset/y_all_v2.npy')

# y_all = utils.to_categorical(y_all, num_classes=3)

# train_ratio = 0.7
# val_ratio = 0.15
# test_ratio = 0.15

# # Divida o conjunto de dados em conjunto de treinamento e teste
# x_train_val, x_test, y_train_val, y_test = train_test_split(x_all, y_all, stratify=y_all, test_size=test_ratio, random_state=123)

# # Divida o conjunto de treinamento em conjunto de treinamento e validação
# x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, stratify=y_train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=123)

# print('CREATING MODEL -----------------------------')

# # Cria o modelo base VGG16
# base_model = VGG16(include_top=False, input_shape=input_shape)

# x = base_model.output
# x = Conv2D(4096, (6, 6), 
#                  activation='relu', 
#                  input_shape=(12, 12, 512), 
#                  strides=(6, 6))(x)
# x = Dropout(0.5)(x)
# x = Conv2D(4096, (1, 1), 
#                  activation='relu', 
#                  strides=(1, 1), 
#                  name='conv_7')(x)
# x = Dropout(0.5)(x)
# x = Conv2D(4096, (1, 1), 
#                  strides=(1, 1), 
#                  name='conv_8')(x)
# x = AveragePooling2D((1, 1), 
#                 strides=(1, 1), 
#                 name='add_pool')(x)
# x = Flatten()(x)
# predictions = Dense(3, activation='softmax')(x)
# model = tf.keras.Model(inputs = base_model.input, outputs = predictions)
# print(model.summary())
# # exit()

# for i, layer in enumerate(model.layers):
#     print(i, layer.name, layer.trainable)

# lr = 1e-4
# optimizer = Adam(lr)

# METRICS = [
#       "accuracy",
#       tf.keras.metrics.TruePositives(name='tp'),
#       tf.keras.metrics.FalsePositives(name='fp'),
#       tf.keras.metrics.TrueNegatives(name='tn'),
#       tf.keras.metrics.FalseNegatives(name='fn'), 
#       tf.keras.metrics.Precision(name='precision'),
#       tf.keras.metrics.Recall(name='recall'),
#       tf.keras.metrics.AUC(name='auc'),
#       tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
# ]

# # Compilar modelo
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer=optimizer,
#     metrics=METRICS,
# )

# # Fit model (storing  weights) -------------------------------------------
# filepath="./weights.hdf5"
# checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, 
#                              monitor='val_accuracy', 
#                              verbose=1, 
#                              save_best_only=True, 
#                              mode='max')

# lr_reduce   = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=1e-5, patience=3, verbose=0)
# early       = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
# checkpoint  = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint, lr_reduce, early]

# print('TRAINING MODEL -----------------------------')

# x_train = np.squeeze(x_train)
# x_val = np.squeeze(x_val)
# x_test = np.squeeze(x_test)

# history = model.fit(x_train, y_train, 
#           validation_data=(x_val, y_val),
#           batch_size=32, 
#           epochs=100, 
#           verbose=1,
#           callbacks=callbacks_list)

# ### storing Model in JSON --------------------------------------------------

# model_json = model.to_json()

# with open("./model.json", "w") as json_file:
#     json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

# ### evaluate model ---------------------------------------------------------

# score = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1]) 
# exit()
### EXTRACT FEATURES ---------------------------------------------------------

cities_indicators = pd.read_csv('../../excel-files/cities_indicators.csv')
model = load_model('./weights.hdf5')
extract_model = tf.keras.Model(model.inputs, model.layers[-3].output) 
print(extract_model.summary())
# exit()
def process_input(img_path):
    try:
        img = utils.load_img('../../' + img_path, target_size=input_shape)
        x = utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        img_features = extract_model.predict(x)[0]
        return img_features
    except NameError:
        print('error => ', img_path)
        print('error => ', NameError)
        return None
    
cities_images = []
population_labels = []
income_labels = []
count = 0
for city in cities_indicators['city_code'].unique():
    df_filter = cities_indicators[cities_indicators['city_code']==city]
    city_images = []
    print(count)
    count = count+1
    for i in range(df_filter.shape[0]):
        img = process_input(df_filter.iloc[i, 0])
        city_images.append(img)
    city_feat = np.append(np.mean(city_images, axis=0), df_filter.iloc[0, 1])
    cities_images.append(city_feat)
    population_labels.append(df_filter.iloc[0, 5])
    income_labels.append(df_filter.iloc[0, 6])

features_final = np.asarray(cities_images)
print(features_final.shape)

features_finetuning = features_final
population_finetuning = np.asarray(population_labels)
income_finetuning = np.asarray(income_labels)

### Save data --------------------------------------------------------------
np.save('../../dataset/features/vgg16_from_scratch/features_with_city_code.npy', features_finetuning)
# np.save('../../dataset/features/vgg16_from_scratch/population_imagenet_finetuning.npy', population_finetuning)
# np.save('../../dataset/features/vgg16_from_scratch/income_imagenet_finetuning.npy', income_finetuning)