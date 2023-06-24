from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

all_features = pd.DataFrame(np.loadtxt('./all_features.csv'))

x_all = all_features.loc[:,:8213]
y_all = all_features.loc[:,8214]
y_all_log = all_features.loc[:,8215]

scaler = StandardScaler()
x_all = scaler.fit_transform(x_all)

pca = PCA()
x_all = pca.fit_transform(x_all)

train_X, test_X, train_y, test_y = train_test_split(x_all, y_all, test_size=0.15, random_state=0)

param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100], 'l1_ratio': [0, .05, .15, .5, .7, .9, .95, .99, 1], 'max_iter': [10000], 'tol': [0.0001]}
grid_cv = GridSearchCV(ElasticNet(), param_grid, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_cv.fit(train_X, train_y)

print('VAL MAE => ', -grid_cv.best_score_)
print('BEST PARAMS => ', grid_cv.best_params_)
print('RESULTS => ', grid_cv.cv_results_)

predictions = grid_cv.predict(test_X)

# Evaluate the model
R2 = r2_score(test_y, predictions)
RMSE = mean_squared_error(test_y, predictions, squared=False)
MAE = mean_absolute_error(test_y, predictions)

print('R2 ', R2)
print('RMSE ', RMSE)
print('MAE ', MAE)