import os
import shutil
import re
import numpy as np
import keras.utils as utils
from keras.applications.imagenet_utils import preprocess_input
import pandas as pd
from sklearn.preprocessing import normalize

# Diretório de origem onde estão as imagens
diretorio_origem = "../dataset/google_images_all"

df = pd.read_csv('../excel-files/nearest_nightlights_per_city.csv')
# Função para extrair a informação da radiancia a partir do nome do arquivo
def split_file_name(file_name):
    split = file_name.split('_')
    return [int(split[0]), float(split[1])]
    
input_shape = (224,224,3)

def process_input(img_path):
    # print(img_path)

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

trainData = []
trainLabels_reg = []

index = 0
for file_name in os.listdir(diretorio_origem):
    full_path = os.path.join(diretorio_origem, file_name)
    
    # Verifique se é um arquivo
    if os.path.isfile(full_path):
        [code, rank] = split_file_name(file_name)
        print(file_name)
        # process input
        a = process_input(diretorio_origem + '/' + file_name)
        if a is not None:
            trainData.append(a)

        data = df[(df['city_code'] == code) & (df['rank'] == rank)]
        # print(data['radiance'].iloc[0])
        # exit()
        trainLabels_reg.append(data['radiance'].iloc[0])
    
    # if (len(trainLabels_reg) > 50):
    #     break
    print(index)
    index = index +1
            
normalized = normalize([trainLabels_reg])[0]
log_transformed = np.log1p(normalized)
print(log_transformed)
x_all_reg = np.asarray(trainData)
y_all_reg_normalized = np.asarray(normalized)
y_all_reg_log = np.asarray(log_transformed)

# ## Save data --------------------------------------------------------------
np.save('../dataset/inputs/x_all_reg.npy', x_all_reg)
np.save('../dataset/inputs/y_all_reg_normalized.npy', y_all_reg_normalized)
np.save('../dataset/inputs/y_all_reg_log.npy', y_all_reg_log)