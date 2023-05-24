from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow.keras.utils as utils
from sklearn.metrics import make_scorer

# Carrega o conjunto de dados
x_all = np.load('../../dataset/features/vgg16_from_scratch/features_with_city_code.npy')
# y_all = np.load('../../dataset/features/vgg16_from_scratch/population_imagenet_finetuning.npy')
y_all = np.load('../../dataset/features/vgg16_from_scratch/income_imagenet_finetuning.npy')

print(x_all.shape)
print(y_all.shape)

# remove city_code column
x_all = np.delete(x_all, -1, axis=1)

# y in natural log
# y_all = np.log(y_all)

# Normalizando os dados
scaler = StandardScaler()
x_all = scaler.fit_transform(x_all)

### TUNE LAMBDA

# Divida o conjunto de dados em conjunto de treinamento e validação
x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.3, random_state=42)

print('X TRAIN SHAPE => ', x_train.shape)
print('Y TRAIN SHAPE => ', x_val.shape)

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
# param_grid = {'alpha': [0.1], 'l1_ratio': [1], 'max_iter': [10000], 'tol': [0.0001]}
# scoring = ['neg_mean_absolute_error', 'r2', 'neg_root_mean_squared_error']
# def rmse(y_true, y_pred, **kwargs):
#     return mean_squared_error(y_true, y_pred, squared=False)
# scorer = make_scorer(rmse, greater_is_better=False)
# scoring = {"r2": make_scorer(r2_score, greater_is_better=True), "mae": make_scorer(mean_absolute_error, greater_is_better=False), "rmse": scorer}
grid_cv = GridSearchCV(ElasticNet(), param_grid, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1)

# Train the model
grid_cv.fit(x_train, y_train)

# Get the best hyperparameters
alpha = grid_cv.best_params_['alpha']
l1_ratio = grid_cv.best_params_['l1_ratio']
# r2 = grid_cv.best_score_['r2']
# mae = grid_cv.best_score_['neg_mean_absolute_error']
# rmse = grid_cv.best_score_['neg_root_mean_squared_error']

print('best alpha => ', alpha)
print('best l1_ratio => ', l1_ratio)
# print('cv results => ', grid_cv.cv_results_)
print('avg score => ', grid_cv.best_score_)
# print('best mae => ', mae)
# print('best rmse => ', rmse)

# Train the model on the complete training set
model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, tol=0.0001)
# model = ElasticNetCV(alpha=0.01, l1_ratio=1, max_iter=10000, tol=0.0001,cv=10)
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


# print('predictions_exp ', predictions_exp)
# print('y_exp ', y_exp)

# R2_exp = r2_score(y_exp, predictions_exp)
# RMSE_exp = mean_squared_error(y_exp, predictions_exp, squared=False)
# MAE_exp = mean_absolute_error(y_exp, predictions_exp)

# print('R2 exp ', R2_exp)
# print('RMSE exp ', RMSE_exp)
# print('MAE exp ', MAE_exp)
