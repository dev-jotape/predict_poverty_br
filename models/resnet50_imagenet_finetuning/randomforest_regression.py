from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA

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
param_grid = {
    'n_estimators': [100, 150, 200, 250, 300],
    # 'max_depth': [1,2,3,4],
}

def rmse(y_true, y_pred, **kwargs):
    RMSE = mean_squared_error(y_true, y_pred, squared=False)
    return RMSE
scorer = make_scorer(rmse, greater_is_better=False)
scoring = {"rmse": scorer, 'mae': 'neg_mean_absolute_error', 'r2': 'r2'}
refit = 'mae'
regr = RandomForestRegressor(random_state=0)
grid_cv = GridSearchCV(regr, param_grid, cv=10, scoring=scoring, n_jobs=-1, refit=refit)

# Train the model
grid_cv.fit(x_train, y_train)

# Get the best hyperparameters
n_estimators = grid_cv.best_params_['n_estimators']
# max_depth = grid_cv.best_params_['max_depth']
rank = np.where(grid_cv.cv_results_['rank_test_'+refit]==1)[0][0]
print('best n_estimators => ', n_estimators)
# print('best max_depth => ', max_depth)
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
best n_estimators =>  300
avg score =>  -120.6430596969697
best position =>  4
mae =>  -120.6430596969697
rmse =>  -171.85562360171508
r2 =>  0.49137442023276934
R2  0.2599475730268008
RMSE  300.2881027537769
MAE  181.50903333333346
total =>  41586.920000000006
tamanho =>  119
media =>  349.4699159663866
desvio absoluto =>  -470.11297566335634
desvio porcentagem =>  -134.52172967832047


'''