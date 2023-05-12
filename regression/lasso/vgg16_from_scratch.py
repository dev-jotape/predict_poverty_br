from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet

# Carrega o conjunto de dados
x_all = np.load('../../dataset/x_all_population_from_scratch.npy')
y_all = np.load('../../dataset/y_all_population_from_scratch.npy')

print(x_all.shape)
print(y_all.shape)

y_all = np.log(y_all)
# Normalizando os dados
# scaler = StandardScaler()
# x_all = scaler.fit_transform(x_all)

### TUNE LAMBDA

# Divida o conjunto de dados em conjunto de treinamento e validação
x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.2, random_state=42)

print(x_train.shape)
print(x_val.shape)

# Set up GridSearchCV with nested cross-validation
param_grid = {'alpha': [0.5, 0.7, 1], 'l1_ratio': [0, 1, 0.5], 'max_iter': [10000], 'tol': [0.0001]}
grid_cv = GridSearchCV(ElasticNet(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Train the model
grid_cv.fit(x_train, y_train)

# Get the best hyperparameters
alpha = grid_cv.best_params_['alpha']
l1_ratio = grid_cv.best_params_['l1_ratio']

print('best alpha => ', alpha)
print('best l1_ratio => ', l1_ratio)

# Train the model on the complete training set
model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000, tol=0.0001)
model.fit(x_train, y_train)

# USANDO LASSO LIB ---------------------------------------------------------
# # Cria um objeto Lasso com alpha=1.0 (parâmetro de regularização)
# model = Lasso(alpha=0.5)

# # Ajusta o modelo aos dados de treinamento
# model.fit(x_train, y_train)

# # Realiza a predição dos dados de teste
# y_pred = model.predict(x_test)
# --------------------------------------------------------------------------

# Make predictions on test set
predictions = model.predict(x_val)

# Evaluate the model
R2 = r2_score(y_val, predictions)
RMSE = mean_squared_error(y_val, predictions, squared=False)
MAE = mean_absolute_error(y_val, predictions)


print('predictions ', predictions)
print('real ', y_val)

print('R2 ', R2)
print('RMSE ', RMSE)
print('MAE ', MAE)
