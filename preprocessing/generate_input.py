import os
import shutil
import re
import numpy as np
import keras.utils as utils
from keras.applications.imagenet_utils import preprocess_input

# Diretório de origem onde estão as imagens
diretorio_origem = "../dataset/google_images_all"

# Função para extrair a informação da radiance a partir do nome do arquivo
def extract_radiance(file_name):
    # Use uma expressão regular para encontrar a radiance no nome do arquivo
    padrao = r'_(\d+)\.png$'
    correspondencia = re.search(padrao, file_name)
    if correspondencia:
        radiance = correspondencia.group(1)
        return radiance
    else:
        return None
    
input_shape = (224,224,3)

def process_input(img_path):
    print(img_path)

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

# Loop através dos arquivos no diretório de origem
radiances = []
for file_name in os.listdir(diretorio_origem):
    full_path = os.path.join(diretorio_origem, file_name)
    
    # Verifique se é um arquivo
    if os.path.isfile(full_path):
        radiance = extract_radiance(file_name)
        
        # Verifique se a radiance foi extraída
        if radiance is not None:
            radiance = int(radiance) / 100
            radiances.append(radiance)

radiances = np.array(radiances)
p25 = np.quantile(radiances, 0.25)
p33 = np.quantile(radiances, 0.3333)
p50 = np.quantile(radiances, 0.5)
p66 = np.quantile(radiances, 0.6666)
p75 = np.quantile(radiances, 0.75)

print(p25)
print(p33)
print(p25)
print(p66)
print(p75)

trainData = []
trainLabels_p3 = []
trainLabels_p4 = []
trainLabels_gmm = []

for file_name in os.listdir(diretorio_origem):
    full_path = os.path.join(diretorio_origem, file_name)
    
    # Verifique se é um arquivo
    if os.path.isfile(full_path):
        radiance = extrair_radiance(file_name)
        
        # Verifique se a radiance foi extraída
        if radiance is not None:
            radiance = int(radiance) / 100

            # process input
            a = process_input(diretorio_origem + '/' + file_name)
            if a is not None:
                trainData.append(a)
            
            # percentil 3
            if radiance < p33:
                trainLabels_p3.append(0)
            elif radiance < p66:
                trainLabels_p3.append(1)
            else:
                trainLabels_p3.append(2)

            # percentil 4
            if radiance < p25:
                trainLabels_p4.append(0)
            elif radiance < p50:
                trainLabels_p4.append(1)
            elif radiance < p75:
                trainLabels_p4.append(2)
            else:
                trainLabels_p4.append(3)

            # gmm
            if radiance < 0.9692701:
                trainLabels_gmm.append(0)
            elif radiance < 4.0585337:
                trainLabels_gmm.append(1)
            else:
                trainLabels_gmm.append(2)

x_all = np.asarray(trainData)
y_all_p3 = np.asarray(trainLabels_p3)
y_all_p4 = np.asarray(trainLabels_p4)
y_all_gmm = np.asarray(trainLabels_gmm)

## Save data --------------------------------------------------------------
np.save('../dataset/inputs/x_all.npy', x_all)
np.save('../dataset/inputs/y_all_p3.npy', y_all_p3)
np.save('../dataset/inputs/y_all_p4.npy', y_all_p4)
np.save('../dataset/inputs/y_all_gmm.npy', y_all_gmm)