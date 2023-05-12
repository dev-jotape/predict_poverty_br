import requests
from os.path import exists
import pandas as pd
from sklearn.mixture import GaussianMixture 
import numpy as np

# Define API parameters ------------------------------------------------------

base_url = "BASE_URL"
key = "KEY"
count = 0

# Functions to download images -----------------------------------------------

def download_image(lat, lon, filename):
    file_exists = exists(filename)
    if not file_exists:
        param_url = "maptype=satellite&center="+ lat + "," + lon + "&zoom=16&size=400x400&style=feature:all|element:labels|visibility:off&format=png&key="
        final_url = base_url + param_url + key
        r = requests.get(final_url)
        if r.status_code == 200:
            with open(filename, 'wb') as img:
                img.write(r.content)
        else:
            print('ERRO TO GET IMAGE ', r.status_code)

def get_classified_images():
    c = 0
    image_folder_url = '../model/google_images/'
    nightlights = pd.read_csv('../model/nearest_nightlights_per_city.csv')

    gmm = GaussianMixture(n_components=3)  # n_components define o número de clusters
    radiance = np.array(nightlights['radiance']).reshape(-1, 1)
    # ajustando o modelo aos dados
    gmm.fit(radiance)

    # prevendo as classes para cada ponto de dados
    labels = gmm.predict(radiance)

    # separando o array em diferentes clusters
    cluster1 = nightlights[labels == 0]
    cluster2 = nightlights[labels == 1]
    cluster3 = nightlights[labels == 2]

    arr = [cluster1.sort_values(by='radiance').iloc[0]['radiance'], cluster2.sort_values(by='radiance').iloc[0]['radiance'], cluster3.sort_values(by='radiance').iloc[0]['radiance']]
    arr = np.sort(arr)

    print(arr)

    imageClass1 = np.where(arr == cluster1.sort_values(by='radiance').iloc[0]['radiance'])[0][0] + 1
    imageClass2 = np.where(arr == cluster2.sort_values(by='radiance').iloc[0]['radiance'])[0][0] + 1
    imageClass3 = np.where(arr == cluster3.sort_values(by='radiance').iloc[0]['radiance'])[0][0] + 1

    print('saving cluster 1 => ', imageClass1)
    for i, row in cluster1.iterrows():
        intensity = row['radiance']
        intensity_file = round(intensity * 100)
        city_code = row['city_code']
        rank = int(row['rank'])
        lat = row['lat']
        lon = row['long']

        className = 'class{}/'.format(imageClass1)

        filename = image_folder_url + className + str(city_code) + '_' + str(rank) + '_' + str(intensity_file) + '.png'
        download_image(str(lat), str(lon), filename)

    print('saving cluster 2 => ', imageClass2)
    for i, row in cluster2.iterrows():
        intensity = row['radiance']
        intensity_file = round(intensity * 100)
        city_code = row['city_code']
        rank = int(row['rank'])
        lat = row['lat']
        lon = row['long']

        className = 'class{}/'.format(imageClass2)

        filename = image_folder_url + className + str(city_code) + '_' + str(rank) + '_' + str(intensity_file) + '.png'
        download_image(str(lat), str(lon), filename)

    print('saving cluster 3 => ', imageClass3)
    for i, row in cluster3.iterrows():
        intensity = row['radiance']
        intensity_file = round(intensity * 100)
        city_code = row['city_code']
        rank = int(row['rank'])
        lat = row['lat']
        lon = row['long']

        className = 'class{}/'.format(imageClass3)

        filename = image_folder_url + className + str(city_code) + '_' + str(rank) + '_' + str(intensity_file) + '.png'
        download_image(str(lat), str(lon), filename)

# Download images ------------------------------------------------------------

get_classified_images()