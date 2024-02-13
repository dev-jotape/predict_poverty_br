from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
import time
import pandas as pd

version = 'p4'

# Carrega o conjunto de dados
x_all = np.load('../../dataset/features/resnet50_imagenet_finetuning/features_{}.npy'.format(version))
y_all = np.load('../../dataset/features/resnet50_imagenet_finetuning/income_{}.npy'.format(version))
# y_all = np.load('../../dataset/features/resnet50_imagenet_finetuning/density_{}.npy'.format(version))
# y_all = np.load('../../dataset/features/resnet50_imagenet_finetuning/gpd_{}.npy'.format(version))
# print(y_all)
# print(x_all.shape)

# exit()

# Normalizando os dados
scaler = StandardScaler()
x_all = scaler.fit_transform(x_all)

#scale predictor variables
# pca = PCA()
# x_all = pca.fit_transform(x_all)

### TUNE LAMBDA

# Divida o conjunto de dados em conjunto de treinamento e validação
x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.15, random_state=0)

# Set up GridSearchCV with nested cross-validation
'''
n_estimators: 
max_depth: 
n_jobs: Numero de jobs rodando em paralelo. None significa 1. -1 significa todos os processadores.
cv: numero de folds. Ex: se 5, os dados serão divididos em 5 folds e o modelo será executado 5 vezes, cada vez com um conjunto diferente.
'''
param_grid = {'kernel' : ('linear', 'poly', 'rbf', 'sigmoid'),'C' : [1,5,10],'degree' : [3,8],'coef0' : [0.01,10,0.5],'gamma' : ('auto','scale')},

modelsvr = SVR(),

def rmse(y_true, y_pred, **kwargs):
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    return RMSE
scorer = make_scorer(rmse, greater_is_better=False)
scoring = {"rmse": scorer, 'mae': 'neg_mean_absolute_error', 'r2': 'r2'}
refit = 'mae'
regr = SVR()
grid_cv = GridSearchCV(regr, param_grid, cv=10, scoring=scoring, n_jobs=-1, refit=refit)

t1 = time.time()
print(t1)
# Train the model
grid_cv.fit(x_train, y_train)
t2 = time.time()
print('time 2 => ', t2)
print(t2 - t1)
# Get the best hyperparameters
kernel = grid_cv.best_params_['kernel']
c = grid_cv.best_params_['C']
degree = grid_cv.best_params_['degree']
coef0 = grid_cv.best_params_['coef0']
gamma = grid_cv.best_params_['gamma']
rank = np.where(grid_cv.cv_results_['rank_test_'+refit]==1)[0][0]
print('best kernel => ', kernel)
print('best c => ', c)
print('best degree => ', degree)
print('best coef0 => ', coef0)
print('best gamma => ', gamma)

print('avg score => ', grid_cv.best_score_)
print('best position => ', rank)
print('mae => ', grid_cv.cv_results_['mean_test_mae'][rank])
print('rmse => ', grid_cv.cv_results_['mean_test_rmse'][rank])
print('r2 => ', grid_cv.cv_results_['mean_test_r2'][rank])

# print('CV RESULTS => ', grid_cv.cv_results_)
df = pd.DataFrame(grid_cv.cv_results_)
df.to_csv('result.csv')

predictions = grid_cv.predict(x_test)

# Evaluate the model
R2 = r2_score(y_test, predictions)
RMSE = mean_squared_error(y_test, predictions, squared=False)
MAE = mean_absolute_error(y_test, predictions)

print('R2 ', R2)
print('RMSE ', RMSE)
print('MAE ', MAE)

# total = np.sum(y_train)
# length = len(y_train)
# avg = total / length
# desvio_absoluto = grid_cv.cv_results_['mean_test_mae'][rank] - avg
# desvio_percentual = (desvio_absoluto/avg) * 100
# print('total => ', total)
# print('tamanho => ', length)
# print('media => ', avg)
# print('desvio absoluto => ', desvio_absoluto)
# print('desvio porcentagem => ', desvio_percentual)


'''
INCOME (P4)
508.9909899234772
best kernel =>  poly
best c =>  1
best degree =>  3
best coef0 =>  10
best gamma =>  scale
avg score =>  -113.67687526929524
best position =>  21
mae =>  -113,67687526929524
rmse =>  -163,21000569052742
r2 =>  0,5960034937519711
R2  0,4473708361304707
RMSE  259,4918859948335
MAE  156,67660167918456
total =>  41586.920000000006
tamanho =>  119
media =>  349.4699159663866
desvio absoluto =>  -463.14679123568186
desvio porcentagem =>  -132.5283722791833

DENSITY (P4)
best kernel =>  sigmoid
best c =>  10
best degree =>  3
best coef0 =>  0.01
best gamma =>  auto
avg score =>  -18.689889542771272
best position =>  99
mae =>  -18,689889542771272
rmse =>  -31,61700388039545
r2 =>  0,1710154305350026
R2  0,10514489394964166
RMSE  42,66301242462116
MAE  27,980682256008176
total =>  3497.03
tamanho =>  119
media =>  29.386806722689077
desvio absoluto =>  -48.07669626546035
desvio porcentagem =>  -163.5995932431172

GPD (P4)
best kernel =>  poly
best c =>  10
best degree =>  3
best coef0 =>  10
best gamma =>  auto
avg score =>  -8813.706078213363
best position =>  113
mae =>  -8813,706078213363
rmse =>  -13371,518328859738
r2 =>  0,17066586763260694
R2  0,294230226565275
RMSE  10608,452625879529
MAE  6679,907878268428
total =>  2744643,4000000004
tamanho =>  119
media =>  23064.230252100842
desvio absoluto =>  -31877.936330314205
desvio porcentagem =>  -138.21374475487016
'''