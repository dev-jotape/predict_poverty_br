from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow.keras.utils as utils

# Carrega o conjunto de dados
# x_all = np.load('../../dataset/features/vgg16_imagenet_finetuning/features_imagenet_finetuning.npy')
x_all = np.load('../../dataset/features/vgg16_imagenet_finetuning/features_with_city_code.npy')
# y_all = np.load('../../dataset/features/vgg16_imagenet_finetuning/population_imagenet_finetuning.npy')
y_all = np.load('../../dataset/features/vgg16_imagenet_finetuning/income_imagenet_finetuning.npy')

print(x_all.shape)
print(y_all.shape)

# remove city_code column
x_all = np.delete(x_all, -1, axis=1)

# y_all = np.log(y_all)

# Normalizando os dados
scaler = StandardScaler()
x_all = scaler.fit_transform(x_all)

### TUNE LAMBDA

# Divida o conjunto de dados em conjunto de treinamento e validação
x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.3, random_state=42)

print(x_train.shape)
print(x_val.shape)

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
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100], 'l1_ratio': [0, 1, 0.5], 'max_iter': [10000], 'tol': [0.0001]}
grid_cv = GridSearchCV(ElasticNet(), param_grid, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1)

# Train the model
grid_cv.fit(x_all, y_all)

# Get the best hyperparameters
alpha = grid_cv.best_params_['alpha']
l1_ratio = grid_cv.best_params_['l1_ratio']

print('best alpha => ', alpha)
print('best l1_ratio => ', l1_ratio)
print('avg score => ', grid_cv.best_score_)

# Train the model on the complete training set
# model = ElasticNet(alpha=10, l1_ratio=1, max_iter=10000, tol=0.0001)
model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, tol=0.0001)
# model = ElasticNet(alpha=1, l1_ratio=0, max_iter=10000, tol=0.0001)
model.fit(x_train, y_train)

# Make predictions on test set
predictions = model.predict(x_val)

# Evaluate the model
R2 = r2_score(y_val, predictions)
RMSE = mean_squared_error(y_val, predictions, squared=False)
MAE = mean_absolute_error(y_val, predictions)

# print('predictions ', predictions)
# print('real ', y_val)
# print('real ', y_ba)

print('R2 ', R2)
print('RMSE ', RMSE)
print('MAE ', MAE)

# if y in log
# predictions_exp = np.exp(predictions)
# y_exp = np.exp(y_val)

# R2_exp = r2_score(y_exp, predictions_exp)
# RMSE_exp = mean_squared_error(y_exp, predictions_exp, squared=False)
# MAE_exp = mean_absolute_error(y_exp, predictions_exp)

# print('R2 exp ', R2_exp)
# print('RMSE exp ', RMSE_exp)
# print('MAE exp ', MAE_exp)
