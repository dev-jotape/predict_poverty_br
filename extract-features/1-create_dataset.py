import os
import numpy as np
from sklearn.model_selection import train_test_split

all_figures = []
trainLabels = []

path_1 = '../model/google_images/class1/'
class_1_files = os.listdir(path_1)
trainLabels += [0] * len(class_1_files)
all_figures += [path_1 + i for i in class_1_files]

path_2 = '../model/google_images/class2/'
class_2_files = os.listdir(path_2)
trainLabels += [1] * len(class_2_files)
all_figures += [path_2 + i for i in class_2_files]

path_3 = '../model/google_images/class3/'
class_3_files = os.listdir(path_3)
trainLabels += [2] * len(class_3_files)
all_figures += [path_3 + i for i in class_3_files]

x_all = np.asarray(all_figures)
y_all = np.asarray(trainLabels)

print(trainLabels)
print(y_all)

# Defina as proporções do conjunto de dados para treinamento, validação e teste
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Divida o conjunto de dados em conjunto de treinamento e teste
x_train_val_data, x_test_data, y_train_val_data, y_test_data = train_test_split(x_all, y_all, stratify=y_all, test_size=test_ratio, random_state=123)

# Divida o conjunto de treinamento em conjunto de treinamento e validação
x_train_data, x_val_data, y_train_data, y_val_data = train_test_split(x_train_val_data, y_train_val_data, stratify=y_train_val_data, test_size=val_ratio/(train_ratio+val_ratio), random_state=123)

# Salve os conjuntos de dados em arquivos separados
np.save('dataset/new/x_train_data.npy',x_train_data)
np.save('dataset/new/y_train_data.npy',y_train_data)
np.save('dataset/new/x_val_data.npy',x_val_data)
np.save('dataset/new/y_val_data.npy',y_val_data)
np.save('dataset/new/x_test_data.npy',x_test_data)
np.save('dataset/new/y_test_data.npy',y_test_data)