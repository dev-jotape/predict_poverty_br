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

# Carrega o conjunto de dados
features = np.load('../../dataset/features/vgg16_imagenet/features_with_city_code.npy')
# y_all = np.load('../../dataset/features/vgg16_imagenet/population.npy')
y_all = np.load('../../dataset/features/vgg16_imagenet/income.npy')
lights = pd.read_csv('../../excel-files/nearest_nightlights_per_city.csv')[['city_code', 'radiance']]
x_df = pd.DataFrame(features)

lights_grouped = lights.groupby(['city_code'], as_index=False).agg({'radiance':['mean','median','max', 'min', 'std']})
lights_grouped.columns = ['city_code', 'mean','median','max', 'min', 'std']

merge = pd.merge(left=x_df, left_on=25088, right=lights_grouped, right_on='city_code', how='left')
merge = merge.loc[:, ~merge.columns.isin([25088, 'city_code'])]
# print(merge)

x_all = merge.to_numpy()

# print(x_all.shape)
# print(y_all.shape)
# exit()
# remove city_code column
# x_all = np.delete(x_all, -1, axis=1)

# y_all = np.log(y_all)

# Normalizando os dados
scaler = StandardScaler()
x_all = scaler.fit_transform(x_all)

### TUNE LAMBDA

# Divida o conjunto de dados em conjunto de treinamento e teste
# x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.15, random_state=42)

# print(x_train.shape)
# print(x_test.shape)

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
# param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100], 'l1_ratio': [0, 1, 0.5, 0.7], 'max_iter': [1000], 'tol': [0.0001]}
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100], 'l1_ratio': [0, .05, .15, .5, .7, .9, .95, .99, 1], 'max_iter': [1000], 'tol': [0.0001]}
grid_cv = GridSearchCV(ElasticNet(), param_grid, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1)

# def mae(y_true, y_pred, **kwargs):
#     print('y_true => ', y_true)
#     print('y_pred => ', y_pred)
#     mae_result = mean_absolute_error(y_true, y_pred)
#     print('MAE RESULT => ', mae_result)
#     print('\n')
#     return mae_result
# scorer = make_scorer(mae, greater_is_better=False)
# scoring = {"mae": scorer}
# param_grid = {'alpha': [1], 'l1_ratio': [0.5], 'max_iter': [10000], 'tol': [0.0001]}
# grid_cv = GridSearchCV(ElasticNet(), param_grid, cv=10, scoring=scoring, n_jobs=-1, refit='mae')

# Train the model
grid_cv.fit(x_all, y_all)

# Get the best hyperparameters
alpha = grid_cv.best_params_['alpha']
l1_ratio = grid_cv.best_params_['l1_ratio']

print('best alpha => ', alpha)
print('best l1_ratio => ', l1_ratio)
print('avg score => ', grid_cv.best_score_)
print('result => ', grid_cv.cv_results_)
exit()
# Train the model on the complete training set
model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, tol=0.0001)
# model = ElasticNet(alpha=1, l1_ratio=0, max_iter=10000, tol=0.0001)
model.fit(x_train, y_train)

# Make predictions on test set
predictions = model.predict(x_test)

# Evaluate the model
R2 = r2_score(y_test, predictions)
RMSE = mean_squared_error(y_test, predictions, squared=False)
MAE = mean_absolute_error(y_test, predictions)

# print('predictions ', predictions)
# print('real ', y_test)

print('R2 ', R2)
print('RMSE ', RMSE)
print('MAE ', MAE)

# if y in log
# predictions_exp = np.exp(predictions)
# y_exp = np.exp(y_test)

# R2_exp = r2_score(y_exp, predictions_exp)
# RMSE_exp = mean_squared_error(y_exp, predictions_exp, squared=False)
# MAE_exp = mean_absolute_error(y_exp, predictions_exp)

# print('R2 exp ', R2_exp)
# print('RMSE exp ', RMSE_exp)
# print('MAE exp ', MAE_exp)
