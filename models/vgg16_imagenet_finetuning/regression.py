from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow.keras.utils as utils
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer

# Carrega o conjunto de dados
x_all = np.load('../../dataset/features/vgg16_imagenet_finetuning/features_with_city_code.npy')
# y_all = np.load('../../dataset/features/vgg16_imagenet_finetuning/income.npy')
y_all = np.load('../../dataset/features/vgg16_imagenet_finetuning/density.npy')

# Normalizando os dados
scaler = StandardScaler()
x_all = scaler.fit_transform(x_all)

#scale predictor variables
# pca = PCA()
# x_all = pca.fit_transform(x_all)

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

# Make predictions on test set
predictions = grid_cv.predict(x_test)

# Evaluate the model
R2 = r2_score(y_test, predictions)
RMSE = mean_squared_error(y_test, predictions, squared=False)
MAE = mean_absolute_error(y_test, predictions)

print('R2 ', R2)
print('RMSE ', RMSE)
print('MAE ', MAE)

'''
best alpha =>  1
best l1_ratio =>  0.95
avg score =>  -127.89982769081591
best position =>  33
mae =>  -127.89982769081591
rmse =>  -183.98389717174848
r2 =>  0.44114029895573187
R2  0.24477036501839133
RMSE  303.3516727130843
MAE  198.6184308322153

----------------------------
DENSITY (WITH PCA)
best alpha =>  100
best l1_ratio =>  0.15
avg score =>  -23.18682734928874
best position =>  47
mae =>  -23.18682734928874
rmse =>  -32.806492116821595
r2 =>  -0.7215018739226877
R2  0.17460514718451603
RMSE  40.97378040004142
MAE  28.89974964575675

DENSITY (WITHOUT PCA)
best alpha =>  10
best l1_ratio =>  0.5
avg score =>  -22.707685735776533
best position =>  39
mae =>  -22.707685735776533
rmse =>  -32.916623721255434
r2 =>  -0.47171981560499104
R2  0.21516903104355933
RMSE  39.95427212356433
MAE  26.89069463945313
'''