from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
import numpy as np
import tensorflow.keras.utils as utils
from sklearn.metrics import make_scorer
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA

# Carrega o conjunto de dados
y_all = np.load('../../dataset/features/vgg16_imagenet/income.npy')

features_image = pd.DataFrame(np.load('../../dataset/features/vgg16_imagenet/features_with_city_code.npy'))
features_finetunin = pd.DataFrame(np.load('../../dataset/features/vgg16_imagenet_finetuning/features_with_city_code_v2.npy'))
basic_features = pd.DataFrame(np.genfromtxt('../../baseline_experiments/google_image_features_basic.csv'))
lights = pd.read_csv('../../excel-files/nearest_nightlights_per_city.csv')[['city_code', 'radiance']]
print(features_finetunin)
print(features_image)
# exit()
basic_features = basic_features.groupby([0], as_index=False).mean()

features_merged = pd.merge(left=features_image, right=features_finetunin, on=4096, how='left')
features_merged = pd.merge(left=features_merged, right=basic_features, left_on=4096, right_on=0, how='left')

lights_grouped = lights.groupby(['city_code'], as_index=False).agg({'radiance':['mean','median','max', 'min', 'std']})
lights_grouped.columns = ['city_code', 'mean','median','max', 'min', 'std']

merge = pd.merge(left=features_merged, left_on=4096, right=lights_grouped, right_on='city_code', how='left')
merge = merge.loc[:, ~merge.columns.isin([4096, 'city_code'])]
merge['income'] = y_all
merge['income_log'] = np.log(y_all)

merge = pd.DataFrame(merge.to_numpy())
# print(merge)
# exit()
x_all = merge.loc[:,:8213]
y_all = merge.loc[:,8214]

print(x_all)

# Normalizando os dados
scaler = StandardScaler()
x_all = scaler.fit_transform(x_all)

#scale predictor variables
pca = PCA()
x_all = pca.fit_transform(x_all)

### TUNE LAMBDA

# Divida o conjunto de dados em conjunto de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.15, random_state=0)

# Set up GridSearchCV with nested cross-validation
'''
l1_ratio: 0 <= l1_ratio <= 1
    0 = ridge regression (L2)
    1 = lasso regression (L1)
alpha: valor da penalização. Se 0, então não há penalização e se enquadra como uma regressão linear simples.
max_iter: numero maximo de iterações permitidas durante o treinamento. Se o algoritmo não convergir após max_iter iterações, ele para e retorna uma mensagem de erro indicando que o modelo não convergiu
tol: diferença mínima aceitável entre o valor da função de custo em duas iterações consecutivas do algoritmo. Se a diferença for menor que a tolerância, o algoritmo é considerado ter convergido e o treinamento é interrompido.
n_jobs: Numero de jobs rodando em paralelo. None significa 1. -1 significa todos os processadores.
cv: numero de folds. Ex: se 5, os dados serão divididos em 5 folds e o modelo será executado 5 vezes, cada vez com um conjunto diferente.
'''
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100], 'l1_ratio': [0, .05, .15, .5, .7, .9, .95, .99, 1], 'max_iter': [10000], 'tol': [0.0001]}

def rmse(y_true, y_pred, **kwargs):
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    return RMSE
scorer = make_scorer(rmse, greater_is_better=False)
scoring = {"rmse": scorer, 'mae': 'neg_mean_absolute_error', 'r2': 'r2'}
refit = 'mae'
grid_cv = GridSearchCV(ElasticNet(), param_grid, cv=10, scoring=scoring, n_jobs=-1, refit=refit)

# Train the model
grid_cv.fit(x_train, y_train)

# Get the best hyperparameters
alpha = grid_cv.best_params_['alpha']
l1_ratio = grid_cv.best_params_['l1_ratio']
rank = np.where(grid_cv.cv_results_['rank_test_'+refit]==1)[0][0]
print('best alpha => ', alpha)
print('best l1_ratio => ', l1_ratio)
print('avg score => ', grid_cv.best_score_)
print('best position => ', rank)
print('mae => ', grid_cv.cv_results_['mean_test_mae'][rank])
print('rmse => ', grid_cv.cv_results_['mean_test_rmse'][rank])
print('r2 => ', grid_cv.cv_results_['mean_test_r2'][rank])
'''
best alpha =>  10
best l1_ratio =>  0.7
avg score =>  -120.74320463979025
best position =>  40
mae =>  -120.74320463979025
rmse =>  -160.55445110503734
r2 =>  0.446389090883675

R2  0.5615922056019922
RMSE  231.12451659954493
MAE  156.28375192706312
'''

# Train the model on the complete training set
# model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, tol=0.0001)
# # model = ElasticNet(alpha=1, l1_ratio=0, max_iter=10000, tol=0.0001)
# model.fit(x_train, y_train)

# # Make predictions on test set
# predictions = model.predict(x_test)
predictions = grid_cv.predict(x_test)

# Evaluate the model
R2 = r2_score(y_test, predictions)
RMSE = mean_squared_error(y_test, predictions, squared=False)
MAE = mean_absolute_error(y_test, predictions)

print('R2 ', R2)
print('RMSE ', RMSE)
print('MAE ', MAE)
