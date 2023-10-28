import os
import shutil
import re
import numpy as np

# Diretório de origem onde estão as imagens
diretorio_origem = "../dataset/google_images_all"

# Função para extrair a informação da radiancia a partir do nome do arquivo
def extrair_radiancia(nome_arquivo):
    # Use uma expressão regular para encontrar a radiancia no nome do arquivo
    padrao = r'_(\d+)\.png$'
    correspondencia = re.search(padrao, nome_arquivo)
    if correspondencia:
        radiancia = correspondencia.group(1)
        return radiancia
    else:
        return None

# Loop através dos arquivos no diretório de origem
radiancias = []
for nome_arquivo in os.listdir(diretorio_origem):
    caminho_completo = os.path.join(diretorio_origem, nome_arquivo)
    
    # Verifique se é um arquivo
    if os.path.isfile(caminho_completo):
        radiancia = extrair_radiancia(nome_arquivo)
        
        # Verifique se a radiancia foi extraída
        if radiancia is not None:
            radiancia = int(radiancia) / 100
            radiancias.append(radiancia)

print(radiancias)
radiancias = np.array(radiancias)
p25 = np.quantile(radiancias, 0.25)
p50 = np.quantile(radiancias, 0.5)
p75 = np.quantile(radiancias, 0.75)

print(p25)
print(p50)
print(p75)

for nome_arquivo in os.listdir(diretorio_origem):
    caminho_completo = os.path.join(diretorio_origem, nome_arquivo)
    
    # Verifique se é um arquivo
    if os.path.isfile(caminho_completo):
        radiancia = extrair_radiancia(nome_arquivo)
        
        # Verifique se a radiancia foi extraída
        if radiancia is not None:
            radiancia = int(radiancia) / 100
            # Defina o diretório de destino com base na radiancia
            if radiancia < p25:
                diretorio_destino = "../dataset/google_images_v2/class1/"
            elif radiancia < p50:
                diretorio_destino = "../dataset/google_images_v2/class2/"
            elif radiancia < p75:
                diretorio_destino = "../dataset/google_images_v2/class3/"
            else:
                diretorio_destino = "../dataset/google_images_v2/class4/"
            # Crie o diretório de destino, se não existir
            os.makedirs(diretorio_destino, exist_ok=True)
            
            # # Mova o arquivo para o diretório de destino
            shutil.copy(caminho_completo, os.path.join(diretorio_destino, nome_arquivo))


