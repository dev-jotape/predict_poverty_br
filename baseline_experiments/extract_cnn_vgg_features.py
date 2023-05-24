# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 22:43:30 2019

@author: diego
"""

### Libraries ---------------------------------------------------------------

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
import pandas as pd
import numpy as np
import tensorflow.keras.utils as utils


### Read income_city_files --------------------------------------------------

# income_city_files = pd.read_csv('../baseline/model/income_city_file.txt')
income_city_files = pd.read_csv('../excel-files/cities_indicators.csv')


### Get CNN features --------------------------------------------------------

base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

features_final = []
pos = 0

for city in income_city_files['city_code'].unique():
    features_temp = []
    pos += 1
    print(pos)
    df_filter = income_city_files[income_city_files['city_code']==city]
    print('city_code => ', df_filter.iloc[0, 1])
    for i in range(df_filter.shape[0]):
        img = utils.load_img('../' + df_filter.iloc[i, 0], target_size=(224, 224))
        x = utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        feat_image = model.predict(x)[0]
        features_temp.append(feat_image)
    city_feat = np.append(np.mean(features_temp, axis=0), df_filter.iloc[0, 1])
    features_final.append(city_feat)
    print(len(features_temp))

features_final = np.asarray(features_final)


### Write features file -----------------------------------------------------

np.savetxt('../dataset/baseline/google_image_features_cnn.csv', features_final)

