import numpy as np
import pandas as pd
from PIL import Image
import pandas as pd
from sklearn.mixture import GaussianMixture 
import numpy as np

from sentinelhub import (
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    bbox_to_dimensions,
)

from sentinelhub import SHConfig

config = SHConfig()
config.sh_client_id='CLIENT_ID'
config.sh_client_secret='CLIENT_SECRET'
config.save()


# The following is not a package. It is a file utils.py which should be in the same folder as this notebook.
from utils import save_image

def download_image(lat, lon, filename):
    bbox = BBox((lon-0.0045, lat-0.0045, lon+0.0045, lat+0.0045), CRS.WGS84)

    evalscript_true_color = """
        //VERSION=3

        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]
                }],
                output: {
                    bands: 3
                }
            };
        }

        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
    """

    request_true_color = SentinelHubRequest(
        evalscript=evalscript_true_color,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=("2021-01-01", "2021-12-30"),
                maxcc=0.001,
                mosaicking_order='leastCC',
                upsampling='BICUBIC',
                downsampling='BICUBIC',
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=bbox,
        size=(400,400),
        config=config,
    )

    true_color_imgs = request_true_color.get_data()

    image = true_color_imgs[0]
    save_image(image, filename, factor=3.5 / 255, clip_range=(0, 1))

def get_classified_images():
    image_folder_url = '../model/sentinel_images/'
    nightlights = pd.read_csv('../model/nearest_nightlights_per_city.csv')

    gmm = GaussianMixture(n_components=3)  # n_components define o número de clusters
    radiance = np.array(nightlights['radiance']).reshape(-1, 1)
    # ajustando o modelo aos dados
    gmm.fit(radiance)

    # prevend
    # o as classes para cada ponto de dados
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

        download_image(lat, lon, filename)

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
        download_image(lat, lon, filename)

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
        download_image(lat, lon, filename)

# Download images ------------------------------------------------------------

get_classified_images()