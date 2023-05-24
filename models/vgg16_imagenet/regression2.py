from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carrega o conjunto de dados
x_all = np.load('../../dataset/features/vgg16_imagenet/features_with_city_code.npy')
# y_all = np.load('../../dataset/features/vgg16_imagenet/population.npy')
y_all = np.load('../../dataset/features/vgg16_imagenet/income.npy')

# remove city_code column
x_all = np.delete(x_all, -1, axis=1)

# y_all = np.log(y_all)

# Normalizando os dados
scaler = StandardScaler()
x_all = scaler.fit_transform(x_all)

# Divida o conjunto de dados em conjunto de treinamento e teste
x_train_all, x_test, y_train_all, y_test = train_test_split(x_all, y_all, test_size=0.15, random_state=42)

# Definindo o número de folds
k = 10

# Lista de valores de lambda (alpha) que você deseja testar
lambda_values = [0.1, 0.5, 1.0, 10]

# Criando o objeto KFold
kf = KFold(n_splits=k, shuffle=True)

lasso_score = {
        '0.1': [],
        '0.5': [],
        '1.0': [],
        '10': [],
    }

# Loop pelos folds
for train_index, val_index in kf.split(x_train_all):
    # Dividindo o conjunto de dados em treino e validacao
    x_train, x_val = x_train_all[train_index], x_train_all[val_index]
    y_train, y_val = y_train_all[train_index], y_train_all[val_index]

    
    # Loop pelos valores de lambda
    for alpha in lambda_values:
        # Criando e treinando o modelo Lasso
        lasso_model = Ridge(alpha=alpha)
        lasso_model.fit(x_train, y_train)
    
        # Fazendo previsões no conjunto de teste usando Lasso
        lasso_y_pred = lasso_model.predict(x_val)
    
        # Calculando o erro médio quadrático (MSE) para Lasso
        lasso_mse = mean_absolute_error(y_val, lasso_y_pred)
        
        lasso_score[str(alpha)].append(lasso_mse)
        # print('temp => ', lasso_score)
        # Imprimindo o MSE para o valor de lambda atual
        # print("MSE para lambda={}: {}".format(alpha, lasso_mse))

print(lasso_score)
print('0.1 => ', sum(lasso_score['0.1']) / len(lasso_score['0.1']))
print('0.5 => ', sum(lasso_score['0.5']) / len(lasso_score['0.5']))
print('1.0 => ', sum(lasso_score['1.0']) / len(lasso_score['1.0']))
print('10 => ', sum(lasso_score['10']) / len(lasso_score['10']))