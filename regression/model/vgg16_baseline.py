import pandas as pd
from keras.models import load_model
import numpy as np
import tensorflow.keras.utils as utils
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from sklearn.preprocessing import StandardScaler

### CREATE MODEL AND EXTRACT FEATURES ---------------------------------------------------
# population = pd.read_csv('../../excel-files/cities_population.csv')

# base_model = load_model('../../extract-features/vgg/weights/weights_baseline.hdf5')
# extract_model = tf.keras.Model(inputs=base_model.inputs, outputs=base_model.get_layer('add_pool').output)
# model_vgg16 = VGG16(weights='imagenet', include_top=False)

# features_final = []
# labels = []
# pos = 0

# for city in population['city_code'].unique():
#     features_temp = []
#     pos += 1
#     print('pos => ', pos)
#     df_filter = population[population['city_code']==city]
#     for i in range(df_filter.shape[0]):
#         img = utils.load_img('../../' + df_filter.iloc[i, 2])
#         x = utils.img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_input(x)
#         feat_image = model_vgg16.predict(x)
#         feat_image_transfer = extract_model.predict(feat_image)[0]
#         features_temp.append(feat_image_transfer)
#     city_feat = np.append(np.mean(features_temp, axis=0), [])
#     features_final.append(city_feat)
#     labels.append(df_filter.iloc[0, 1])

# features_final = np.asarray(features_final)

# x_all = features_final
# y_all = np.asarray(labels)

# np.save('../../dataset/x_all_population_baseline.npy', x_all)
# np.save('../../dataset/y_all_population_baseline.npy', y_all)

x_all = np.load('../../dataset/x_all_population_baseline.npy')
y_all = np.load('../../dataset/y_all_population_baseline.npy')

y_all = np.log(y_all)

# Normalizando os dados
scaler = StandardScaler()
x_all = scaler.fit_transform(x_all)

### CREATE MODEL TO REGRESSION -----------------------------------------------------

train_ratio = 0.8
val_ratio = 0.10
test_ratio = 0.10

# Divida o conjunto de dados em conjunto de treinamento e teste
x_train_val, x_test, y_train_val, y_test = train_test_split(x_all, y_all, test_size=test_ratio, random_state=42)

# # Divida o conjunto de treinamento em conjunto de treinamento e validação
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)

### CREATE MODEL ---------------------------------------------------
lr_reduce   = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.1, min_delta=1e-5, patience=3, verbose=0)
early       = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=10, mode='max')
callbacks_list = [lr_reduce, early]

model = tf.keras.Sequential()
model.add(Dense(1024, input_dim=4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(1, activation='linear'))

opt = Adam(lr=1e-3)
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