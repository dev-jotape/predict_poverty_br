import pandas as pd
from keras.models import load_model
import numpy as np
import tensorflow.keras.utils as utils
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from tensorflow.keras.layers import Dropout, Dense
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

### EXTRACT FEATURES ---------------------------------------------------
population = pd.read_csv('../../excel-files/cities_population.csv')

base_model = load_model('../../extract-features/vgg/weights/weights_from_scratch.hdf5')
extract_model = tf.keras.Model(base_model.inputs, base_model.layers[-5].output) 

input_shape = (224,224,3)

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
labels = []
count = 0
for city in population['city_code'].unique():
    df_filter = population[population['city_code']==city]
    city_images = []
    print(count)
    count = count+1
    for i in range(df_filter.shape[0]):
        img = process_input(df_filter.iloc[i, 2])
        city_images.append(img)
    # city_feat = np.append(np.mean(city_images, axis=0), df_filter.iloc[0, 0])
    city_feat = np.append(np.mean(city_images, axis=0), [])
    cities_images.append(city_feat)
    # cities_images.append(np.mean(city_images, axis=0))
    labels.append(df_filter.iloc[0, 1])

features_final = np.asarray(cities_images)
# features shape 1 =>  (1, 7, 7, 512)
# SHAPE =>  (1, 25088)
print(features_final.shape)

x_all = features_final
y_all = np.asarray(labels)

### Save data --------------------------------------------------------------
np.save('../../dataset/x_all_population_from_scratch.npy', x_all)
np.save('../../dataset/y_all_population_from_scratch.npy', y_all)

# x_all = np.load('../../dataset/x_all_population_from_scratch.npy')
# y_all = np.load('../../dataset/y_all_population_from_scratch.npy')
y_all = np.log(y_all)

# Normalizando os dados
# scaler = StandardScaler()
# x_all = scaler.fit_transform(x_all)

train_ratio = 0.8
val_ratio = 0.10
test_ratio = 0.10

# Divida o conjunto de dados em conjunto de treinamento e teste
x_train_val, x_test, y_train_val, y_test = train_test_split(x_all, y_all, test_size=test_ratio, random_state=42)

# # Divida o conjunto de treinamento em conjunto de treinamento e validação
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)

# print(x_train.shape)

### CREATE MODEL ---------------------------------------------------
lr_reduce   = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.1, min_delta=1e-5, patience=3, verbose=0)
early       = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=10, mode='max')
callbacks_list = [lr_reduce, early]


model = tf.keras.Sequential()
model.add(Dense(256, input_dim=25088, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

opt = Adam(lr=1e-2)
model.compile(optimizer=opt, loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])

# print(model.summary())

# Treinar modelo
model.fit(
    x_train, 
    y_train, 
    batch_size=32, 
    epochs=100, 
    validation_data=(x_val, y_val),
    verbose=1,
    callbacks=callbacks_list
)

score = model.evaluate(x_test, y_test)
print('Erro médio quadrático:', score)


prediction1 = model.predict(x_test)
print('prediction1 => ', prediction1)
print('real => ', y_test)