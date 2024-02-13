from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow.keras.utils as utils
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
import pandas as pd

# version = 'reg_normalized'
version = 'p4'
# Carrega o conjunto de dados
x_all = np.load('../../dataset/features/resnet50_imagenet_finetuning/features_{}.npy'.format(version))
y_all = np.load('../../dataset/features/resnet50_imagenet_finetuning/income_{}.npy'.format(version))
# y_all = np.load('../../dataset/features/resnet50_imagenet_finetuning/density.npy')
# y_all = np.load('../../dataset/features/resnet50_imagenet_finetuning/gpd_{}.npy'.format(version))
print(y_all)
# print(x_all.shape)

# exit()

# Normalizando os dados
scaler = StandardScaler()
x_all = scaler.fit_transform(x_all)

# scaler2 = StandardScaler()
# y_all = scaler2.fit_transform(y_all.reshape(-1,1))

#scale predictor variables
# pca = PCA()
# x_all = pca.fit_transform(x_all)

### TUNE LAMBDA

# Divida o conjunto de dados em conjunto de treinamento e validação
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
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100], 'l1_ratio': [0, .05, .15, .5, .7, .9, 1], 'max_iter': [1000], 'tol': [0.0001]}
# param_grid = {'alpha': [0.1], 'l1_ratio': [1], 'max_iter': [1000], 'tol': [0.0001]}

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

# print('CV RESULTS => ', grid_cv.cv_results_)

predictions = grid_cv.predict(x_test)

df = pd.DataFrame(grid_cv.cv_results_)
df.to_csv('result_elasticnet.csv')

# Evaluate the model
R2 = r2_score(y_test, predictions)
RMSE = mean_squared_error(y_test, predictions, squared=False)
MAE = mean_absolute_error(y_test, predictions)

print('R2 ', R2)
print('RMSE ', RMSE)
print('MAE ', MAE)

print('\n\n')

# predictions_reverse = scaler2.inverse_transform(predictions.reshape(-1,1))
# y_test_reverse = scaler2.inverse_transform(y_test)

# # Evaluate the model
# R2 = r2_score(y_test_reverse, predictions_reverse)
# RMSE = mean_squared_error(y_test_reverse, predictions_reverse, squared=False)
# MAE = mean_absolute_error(y_test_reverse, predictions_reverse)

# print('R2 ', R2)
# print('RMSE ', RMSE)
# print('MAE ', MAE)


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
total =>  41586.920000000006
tamanho =>  119
media =>  349.4699159663866
desvio absoluto =>  -229.6699159663866
desvio porcentagem =>  -65.71950988435788
'''


'''
COM PCA
best alpha =>  1
best l1_ratio =>  0
avg score =>  -125.28525423943344
best position =>  27
mae =>  -125.28525423943344
rmse =>  -167.10634261473328
r2 =>  0.48585042257972927
R2  0.6313389262146458
RMSE  211.94370611715493
MAE  141.34641094433036

SEM PCA
best alpha =>  0.1
best l1_ratio =>  0.05
avg score =>  -119.80087858747432
best position =>  15
mae =>  -119.80087858747432
rmse =>  -165.72463176401334
r2 =>  0.4887727888481347
R2  0.5187656777550106
RMSE  242.15039372892244
MAE  173.35162158801108

--------------------------------------

INCOME (GMM)
best alpha =>  0.1
best l1_ratio =>  0.7
avg score =>  -122.13775479708879
best position =>  18
mae =>  -122.13775479708879
rmse =>  -168.54335126081497
r2 =>  0.49215914970803754
R2  0.5883910215170387
RMSE  223.94909006479477
MAE  156.80783811155644
total =>  41586.920000000006
tamanho =>  119
media =>  349.4699159663866
desvio absoluto =>  -471.6076707634754
desvio porcentagem =>  -134.94943318921807

INCOME (P3)
best alpha =>  0.1
best l1_ratio =>  1
avg score =>  -119.65795033094173
best position =>  20
mae =>  -119.65795033094173
rmse =>  -166.74410788674024
r2 =>  0.4890182590458589
R2  0.3916590662137992
RMSE  272.2578448959787
MAE  182.9808350304119
total =>  41586.920000000006
tamanho =>  119
media =>  349.4699159663866
desvio absoluto =>  -469.12786629732835
desvio porcentagem =>  -134.23984293470653

INCOME (P4)
best alpha =>  1
best l1_ratio =>  0.5
avg score =>  -120.04207464508423
best position =>  24
mae =>  -120.04207464508423
rmse =>  -168.09046935230822
r2 =>  0.5434940906967322
R2  0.5890793172007436
RMSE  223.76176703017498
MAE  142.44656563287003
total =>  41586.920000000006
tamanho =>  119
media =>  349.4699159663866
desvio absoluto =>  -469.5119906114709
desvio porcentagem =>  -134.34975920978286

INCOME (REGRESSION)
best alpha =>  10
best l1_ratio =>  0.05
avg score =>  -141.8590019625239
best position =>  29
mae =>  -141.8590019625239
rmse =>  -188.65023685304604
r2 =>  0.36519369183538974
R2  0.5021197296576648
RMSE  246.30279534681517
MAE  179.84191446171812
total =>  41586.920000000006
tamanho =>  119
media =>  349.4699159663866
desvio absoluto =>  -491.3289179289105
desvio porcentagem =>  -140.59262199157894

INCOME (REGRESSION WITH PCA)
best alpha =>  100
best l1_ratio =>  0.5
avg score =>  -137.207479335208
best position =>  38
mae =>  -137.207479335208
rmse =>  -184.51384907430705
r2 =>  0.3462691458641774
R2  0.4844938728844017
RMSE  250.6246584932309
MAE  182.66875635352002
total =>  41586.920000000006
tamanho =>  119
media =>  349.4699159663866
desvio absoluto =>  -486.67739530159463

--------------------------------------

DENSITY (WITH PCA)
best alpha =>  100
best l1_ratio =>  0
avg score =>  -21.76296329384929
best position =>  35
mae =>  -21.76296329384929
rmse =>  -31.697693804033985
r2 =>  -0.5130110471958467
R2  0.4814196527132364
RMSE  32.47754434787088
MAE  25.241472566777393
total =>  3497.03
tamanho =>  119
media =>  29.386806722689077
desvio absoluto =>  -51.14977001653837
desvio porcentagem =>  -174.05691778360682

DENSITY (WITHOUT PCA)
best alpha =>  1
best l1_ratio =>  0.15
avg score =>  -20.831723459668122
best position =>  23
mae =>  -20.831723459668122
rmse =>  -31.251523799678967
r2 =>  -0.33618562408122193
R2  0.4577584933900791
RMSE  33.21020347254983
MAE  25.90941098573642
total =>  3497.03
tamanho =>  119
media =>  29.386806722689077
desvio absoluto =>  -50.2185301823572
desvio porcentagem =>  -170.88801330559093

DENSITY (P3)
best alpha =>  100
best l1_ratio =>  0.05
avg score =>  -21.542770743670893
best position =>  36
mae =>  -21.542770743670893
rmse =>  -32.83304440102971
r2 =>  -0.37863227087702833
R2  0.44896370924077433
RMSE  33.478443505317045
MAE  24.77196946202814
total =>  3497.03
tamanho =>  119
media =>  29.386806722689077
desvio absoluto =>  -50.92957746635997
desvio porcentagem =>  -173.3076272864927

DENSITY (P4)
best alpha =>  1
best l1_ratio =>  1
avg score =>  -19.966433841174357
best position =>  27
mae =>  -19.966433841174357
rmse =>  -30.680433699620323
r2 =>  -0.159515259993733
R2  0.4264176345557076
RMSE  34.15647564540918
MAE  25.063588022968766
total =>  3497.03
tamanho =>  119
media =>  29.386806722689077
desvio absoluto =>  -49.35324056386344
desvio porcentagem =>  -167.94352999830565

GPD (P4)
best alpha =>  100
best l1_ratio =>  0.9
avg score =>  -9221.76851024749
best position =>  40
mae =>  -9221,76851024749
rmse =>  -12560,690465201606
r2 =>  0,22849209932314926
R2  0,3568359347250918
RMSE  10127,013754245547
MAE  7660,471587421155
total =>  2744643.4000000004
tamanho =>  119
media =>  23064.230252100842
desvio absoluto =>  -32285.998762348332
desvio porcentagem =>  -139.982988417346

GPD (GMM)
best alpha =>  100
best l1_ratio =>  0.05
avg score =>  -9670.86779100216
best position =>  36
mae =>  -9670.86779100216
rmse =>  -13314.388307412813
r2 =>  0.113192648939291
R2  0.4122665787901858
RMSE  9680.787958133722
MAE  7608.7889185791355
total =>  2744643.4000000004
tamanho =>  119
media =>  23064.230252100842
desvio absoluto =>  -32735.098043103004
desvio porcentagem =>  -141.93015628657832
'''