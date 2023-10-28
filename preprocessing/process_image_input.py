import os
import time
import keras.utils as utils
import numpy as np
from keras.applications.imagenet_utils import preprocess_input

input_shape = (224,224,3)

def process_input(img_path):
    try:
        img = utils.load_img(img_path, target_size=input_shape)
        x = utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x
    except NameError:
        print('error => ', img_path)
        print('error => ', NameError)
        return None


# ### Append correct label to data -------------------------------------------

all_figures = []
trainLabels = []

path_1 = '../dataset/google_images_v2/class1/'
class_1_files = os.listdir(path_1)
trainLabels += [0] * len(class_1_files)
all_figures += [path_1 + i for i in class_1_files]

path_2 = '../dataset/google_images_v2/class2/'
class_2_files = os.listdir(path_2)
trainLabels += [1] * len(class_2_files)
all_figures += [path_2 + i for i in class_2_files]

path_3 = '../dataset/google_images_v2/class3/'
class_3_files = os.listdir(path_3)
trainLabels += [2] * len(class_3_files)
all_figures += [path_3 + i for i in class_3_files]

path_4 = '../dataset/google_images_v2/class4/'
class_4_files = os.listdir(path_4)
trainLabels += [3] * len(class_4_files)
all_figures += [path_4 + i for i in class_4_files]

trainData = []

trainData = []
t1 = time.time()
for idx, i in enumerate(all_figures):
    print(idx)
    a = process_input(i)
    if a is not None:
        trainData.append(a)


x_all = np.asarray(trainData)
y_all = np.asarray(trainLabels)

## Save data --------------------------------------------------------------
np.save('../dataset/x_all_v2.npy', x_all)
np.save('../dataset/y_all_v2.npy', y_all)