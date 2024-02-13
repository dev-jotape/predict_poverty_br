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

version = 'p4'

# Carrega o conjunto de dados
x_all = np.load('../../dataset/features/vgg16_imagenet_finetuning/features_{}.npy'.format(version))
# y_all = np.load('../../dataset/features/vgg16_imagenet_finetuning/income_{}.npy'.format(version))
# y_all = np.load('../../dataset/features/vgg16_imagenet_finetuning/density_{}.npy'.format(version))
y_all = np.load('../../dataset/features/vgg16_imagenet_finetuning/gpd_{}.npy'.format(version))
print(y_all)
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

predictions = grid_cv.predict(x_test)

# Evaluate the model
R2 = r2_score(y_test, predictions)
RMSE = mean_squared_error(y_test, predictions, squared=False)
MAE = mean_absolute_error(y_test, predictions)

print('R2 ', R2)
print('RMSE ', RMSE)
print('MAE ', MAE)

total = np.sum(y_train)
length = len(y_train)
avg = total / length
desvio_absoluto = grid_cv.cv_results_['mean_test_mae'][rank] - avg
desvio_percentual = (desvio_absoluto/avg) * 100
print('total => ', total)
print('tamanho => ', length)
print('media => ', avg)
print('desvio absoluto => ', desvio_absoluto)
print('desvio porcentagem => ', desvio_percentual)


'''
INCOME (P4)
best kernel =>  poly
best c =>  5
best degree =>  3
best coef0 =>  10
best gamma =>  auto
avg score =>  -144.41490320736065
best position =>  65
mae =>  -144,41490320736065
rmse =>  -204,30185457316284
r2 =>  0,3528351543324716
R2  0,18210474738502047
RMSE  315,6862935439508
MAE  213,75322855115016
total =>  41586.920000000006
tamanho =>  119
media =>  349.4699159663866
desvio absoluto =>  -493.88481917374725
desvio porcentagem =>  -141.3239871615304

DENSITY (P4)
best kernel =>  sigmoid
best c =>  10
best degree =>  3
best coef0 =>  0.01
best gamma =>  scale
avg score =>  -20.630169844074295
best position =>  103
mae =>  -20,630169844074295
rmse =>  -33,31682426468812
r2 =>  -0,07296178391009236
R2  -0,023047835867730848
RMSE  45,61662263939757
MAE  28,238211989457724
total =>  3497.03
tamanho =>  119
media =>  29.386806722689077
desvio absoluto =>  -50.016976566763375
desvio porcentagem =>  -170.20214900772487

GPD (P3)
best kernel =>  linear
best c =>  1
best degree =>  3
best coef0 =>  0.01
best gamma =>  auto
avg score =>  -9284.732159881736
best position =>  0
mae =>  -9284,732159881736
rmse =>  -14043,221587972428
r2 =>  0,06484142170315343
R2  0,08975637883873311
RMSE  12047,566608336443
MAE  8651,543852929824
total =>  2744643.4000000004
tamanho =>  119
media =>  23064.230252100842
desvio absoluto =>  -32348.96241198258
desvio porcentagem =>  -140.25598105116046
'''