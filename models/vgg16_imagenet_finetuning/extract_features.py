import tensorflow as tf
import keras.utils as utils
from keras.applications.imagenet_utils import preprocess_input
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers.core import Activation, Dense, Flatten, Dropout
import numpy as np
from keras.optimizers import Adam
import json as simplejson
import os
from scikitplot.metrics import plot_confusion_matrix, plot_roc
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
from keras.models import load_model
from keras.layers.convolutional import Conv2D, AveragePooling2D

# import time

print('IMPORT DATA -----------------------------')
input_shape = (224,224,3)

### Load data --------------------------------------------------------------
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
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
# print(base_model.summary())

# model = tf.keras.Sequential([
#     base_model,
#     Conv2D(4096, (6, 6), 
#         activation='relu', 
#         input_shape=(12, 12, 512), 
#         strides=(6, 6), 
#         name='input'),
#     Dropout(0.5),
#     Conv2D(4096, (1, 1), 
#         activation='relu', 
#         strides=(1, 1), 
#         name='conv_7'),
#     Dropout(0.5),
#     Conv2D(4096, (1, 1), 
#         strides=(1, 1), 
#         name='conv_8'),
#     # AveragePooling2D((2, 2), 
#     #     strides=(1, 1), 
#     #     name='add_pool'),
#     Flatten(name='flatten'),
#     Dense(3),
#     Activation("softmax")
# ])

# for i, layer in enumerate(base_model.layers):
#     print(i, layer.name, layer.trainable)

# # x = base_model.get_layer('fc2').output
# # # x = Flatten()(x)
# # x = Dense(256, activation='relu')(x)
# # x = Dropout(0.5)(x)
# # predictions = Dense(3, activation='softmax')(x)
# # model = tf.keras.Model(inputs = base_model.input, outputs = predictions)

# lr = 1e-4
# optimizer = Adam(learning_rate=lr)

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
# filepath="./weights_v3.hdf5"
# checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, 
#                              monitor='val_accuracy', 
#                              verbose=1, 
#                              save_best_only=True, 
#                              mode='max')

# lr_reduce   = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=1e-5, patience=3, verbose=0)
# early       = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
# checkpoint  = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# class PerformanceVisualizationCallback(tf.keras.callbacks.Callback):
#     def __init__(self, model, validation_data, image_dir):
#         super().__init__()
#         self.model = model
#         self.validation_data = validation_data

#         os.makedirs(image_dir, exist_ok=True)
#         self.image_dir = image_dir
#     def on_epoch_end(self, epoch, logs={}):
#         y_pred = np.asarray(self.model.predict(self.validation_data[0]))
#         y_true = np.argmax(self.validation_data[1], axis=1)
#         y_pred_class = np.argmax(y_pred, axis=1)
#         # plot and save confusion matrix
#         fig, ax = plt.subplots(figsize=(16,12))
#         plot_confusion_matrix(y_true, y_pred_class, ax=ax)
#         fig.savefig(os.path.join(self.image_dir, f'confusion_matrix_epoch_{epoch}'))

#         report = classification_report(y_true, y_pred_class, digits=3)
#         text_file = open(os.path.join(self.image_dir, f'report_epoch_{epoch}'), "wt")
#         text_file.write(report)

#         # plot and save roc curve
#         fig, ax = plt.subplots(figsize=(16,12))
#         plot_roc(y_true, y_pred, ax=ax)
#         fig.savefig(os.path.join(self.image_dir, f'roc_curve_epoch_{epoch}'))

# x_train = np.squeeze(x_train)
# x_val = np.squeeze(x_val)
# x_test = np.squeeze(x_test)

# # performance_cbk = PerformanceVisualizationCallback(x_val, y_val, 'confusion_matrix.png')
# performance_cbk = PerformanceVisualizationCallback(
#                       model=model,
#                       validation_data=(x_val, y_val),
#                       image_dir='performance_vizualizations')
# # callbacks_list = [performance_cbk, checkpoint, lr_reduce, early]
# callbacks_list = [checkpoint, lr_reduce, early]

# print('TRAINING MODEL -----------------------------')

# history = model.fit(x_train, y_train, 
#           validation_data=(x_val, y_val),
#           batch_size=32, 
#           epochs=100, 
#           verbose=1,
#           callbacks=callbacks_list)

# ### storing Model in JSON --------------------------------------------------

# model_json = model.to_json()

# with open("./model_v3.json", "w") as json_file:
#     json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

# ### evaluate model ---------------------------------------------------------

# score = model.evaluate(x_test, y_test, verbose=1)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1]) 

### EXTRACT FEATURES ---------------------------------------------------
# cities_indicators = pd.read_csv('../../excel-files/cities_indicators.csv')

# base_model = load_model('./weights_v2.hdf5')
# extract_model = tf.keras.Model(base_model.inputs, base_model.get_layer('conv_8').output) 

base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

print(base_model.summary())
exit()
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
np.save('../../dataset/features/vgg16_imagenet_finetuning/features_with_city_code_v2.npy', features_finetuning)
# np.save('../../dataset/features/vgg16_imagenet_finetuning/population_imagenet_finetuning.npy', population_finetuning)
# np.save('../../dataset/features/vgg16_imagenet_finetuning/income_imagenet_finetuning.npy', income_finetuning)