import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow.keras.utils as utils
from sklearn.model_selection import train_test_split
import json as simplejson
import time
import pandas as pd
from keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import os

# Define the input shape of your model
input_shape = (224, 224, 3)
version = 'reg_normalized'

# Create an instance of the ResNet50 model without the top layer
# base_model = ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')

# for i, layer in enumerate(base_model.layers):
#     print(i, layer.name, layer.trainable)

# # Create a new model instance with the top layer
# x = base_model.output
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(256, activation='relu')(x)
# x = tf.keras.layers.Dropout(0.5)(x)
# predictions = tf.keras.layers.Dense(1, activation='linear')(x)
# model = tf.keras.Model(inputs = base_model.input, outputs = predictions)

# lr = 1e-4
# optimizer = Adam(learning_rate=lr)

# # Compilar modelo
# model.compile(
#     loss='mae',
#     optimizer=optimizer,
#     metrics=['mse', 'mae'],
# )

# # Fit model (storing  weights) -------------------------------------------
# filepath="./weights_{}.hdf5".format(version)
# print('filepath => ', filepath)
# checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, 
#                              monitor='val_mae', 
#                              verbose=1, 
#                              save_best_only=True, 
#                              mode='max')

# lr_reduce   = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mae', factor=0.1, min_delta=1e-5, patience=3, verbose=0)
# early       = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=5, mode='max')
# checkpoint  = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_mae', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint, lr_reduce, early]

# print('IMPORT DATA -----------------------------')

# x_all = np.load('../../dataset/inputs/x_all_reg.npy')
# y_all = np.load('../../dataset/inputs/y_all_{}.npy'.format(version))

# # print('y_all => ', y_all)

# train_ratio = 0.7
# val_ratio = 0.15
# test_ratio = 0.15

# # Divida o conjunto de dados em conjunto de treinamento e teste
# x_train_val, x_test, y_train_val, y_test = train_test_split(x_all, y_all, test_size=test_ratio, random_state=123)

# # Divida o conjunto de treinamento em conjunto de treinamento e validação
# x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=123)

# print('train size => ', len(x_train))
# print('val size => ', len(x_val))
# print('test size => ', len(x_test))

# print('TRAINING MODEL -----------------------------')

# x_train = np.squeeze(x_train)
# x_val = np.squeeze(x_val)
# x_test = np.squeeze(x_test)

# t1 = time.time()

# history = model.fit(x_train, y_train, 
#           validation_data=(x_val, y_val),
#           batch_size=32, 
#           epochs=100, 
#           verbose=1,
#           callbacks=callbacks_list)


# t2 = time.time()
# print('time 1 => ', t1)
# print('time 2 => ', t2)
# print(t2 - t1)

# ### storing Model in JSON --------------------------------------------------

# model_json = model.to_json()

# with open("./model_{}.json".format(version), "w") as json_file:
#     json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))


# ### evaluate model ---------------------------------------------------------

# score = model.evaluate(x_test, y_test, verbose=1)
# print('Test mae:', score[0])
# print('Test mse:', score[1])

# Test mae: 0.0024707880802452564
# Test mse: 3.695721170515753e-05

### EXTRACT FEATURES ---------------------------------------------------------

cities_indicators = pd.read_csv('../../excel-files/cities_indicators.csv')
model = load_model("./weights_{}.hdf5".format(version))
extract_model = tf.keras.Model(model.inputs, model.layers[-5].output) 

print(extract_model.summary())

def process_input(img_path):
    img = os.path.basename(img_path)
    try:
        img = utils.load_img('../../dataset/google_images_all/' + img, target_size=input_shape)
        x = utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        predicted = extract_model.predict(x)[0]
        # print('predicted => ', predicted.shape)
        # print('predicted[0] => ', predicted[0].shape)
        img_features = predicted
        return img_features
    except NameError:
        print('error => ', img)
        print('error => ', NameError)
        return None
    
cities_images = []
population_labels = []
income_labels = []
density_labels = []
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
    density_labels.append(df_filter.iloc[0, 8])

features_final = np.asarray(cities_images)

features_finetuning = features_final
population_finetuning = np.asarray(population_labels)
income_finetuning = np.asarray(income_labels)
density_finetuning = np.asarray(density_labels)

# print(density_finetuning)

### Save data --------------------------------------------------------------
np.save('../../dataset/features/resnet50_imagenet_finetuning/features_{}.npy'.format(version), features_finetuning)
np.save('../../dataset/features/resnet50_imagenet_finetuning/population_{}.npy'.format(version), population_finetuning)
np.save('../../dataset/features/resnet50_imagenet_finetuning/income_{}.npy'.format(version), income_finetuning)
np.save('../../dataset/features/resnet50_imagenet_finetuning/density_{}.npy'.format(version), density_finetuning)