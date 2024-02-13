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

# Carrega o conjunto de dados
x_all = np.load('../../dataset/features/resnet50_resisc45/features_with_city_code.npy')
# y_all = np.load('../../dataset/features/resnet50_resisc45/income.npy')
# y_all = np.load('../../dataset/features/resnet50_imagenet_finetuning/density.npy')
y_all = np.load('../../dataset/features/resnet50_imagenet_finetuning/gpd_gmm.npy')

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

predictions = grid_cv.predict(x_test)

# Evaluate the model
R2 = r2_score(y_test, predictions)
RMSE = mean_squared_error(y_test, predictions, squared=False)
MAE = mean_absolute_error(y_test, predictions)

print('R2 ', R2)
print('RMSE ', RMSE)
print('MAE ', MAE)

'''
best alpha =>  10
best l1_ratio =>  1
avg score =>  -167.90321814869517
best position =>  34
mae =>  -167.90321814869517
rmse =>  -224.30951466339997
r2 =>  0.13664152209288694
R2  0.34873058974681725
RMSE  281.7002435565606
MAE  192.5803893602572

DENSITY
best alpha =>  10
best l1_ratio =>  0.5
avg score =>  -26.669801945550425
best position =>  31
mae =>  -26.669801945550425
rmse =>  -37.18474105400786
r2 =>  -0.6351054460306521
R2  0.03885618191897411
RMSE  44.21497113673107
MAE  31.849086345492744

GPD
best alpha =>  100
best l1_ratio =>  0.7
avg score =>  -11213.800971728488
best position =>  39
mae =>  -11213.800971728488
rmse =>  -14748.093629077726
r2 =>  -0.07788027577217019
R2  0.22936302242605
RMSE  11085.248571423568
MAE  8979.165562331864
'''