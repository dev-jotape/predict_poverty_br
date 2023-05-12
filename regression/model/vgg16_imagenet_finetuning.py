import pandas as pd
from keras.models import load_model
import numpy as np
import tensorflow.keras.utils as utils
from keras.applications.imagenet_utils import preprocess_input
from keras.optimizers import Adam
from keras.losses import MeanSquaredError
from keras.metrics import MeanAbsoluteError
from tensorflow.keras.layers import Dropout, Dense
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Carrega o conjunto de dados
x_all = np.load('../../dataset/features/features_imagenet_finetuning.npy')
# y_all = np.load('../../dataset/features/population_imagenet_finetuning.npy')
y_all = np.load('../../dataset/features/income_imagenet_finetuning.npy')
y_all = np.log(y_all)

# Normalizando os dados
scaler = StandardScaler()
x_all = scaler.fit_transform(x_all)

# train_ratio = 0.7
# val_ratio = 0.15
# test_ratio = 0.15

# # Divida o conjunto de dados em conjunto de treinamento e teste
# x_train_val, x_test, y_train_val, y_test = train_test_split(x_all, y_all, test_size=test_ratio, random_state=42)

# # # Divida o conjunto de treinamento em conjunto de treinamento e validação
# x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size=0.2, random_state=32)

### CREATE MODEL ---------------------------------------------------
# lr_reduce   = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_mean_absolute_error', factor=0.1, min_delta=1e-5, patience=3, verbose=0)
# early       = tf.keras.callbacks.EarlyStopping(monitor='val_mean_absolute_error', patience=10, mode='max')
# callbacks_list = [lr_reduce, early]


# model = tf.keras.Sequential()
# model.add(Dense(256, input_dim=25088, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='linear'))

# opt = Adam(learning_rate=1e-2)
# model.compile(optimizer=opt, loss=MeanSquaredError(), metrics=[MeanAbsoluteError()])

def build_model():
  model = tf.keras.Sequential([
    Dense(64, activation='relu', input_dim=25088),
    Dense(64, activation='relu'),
    Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

print(model.summary())

EPOCHS = 1000

# Treinar modelo
model.fit(
    x_train, 
    y_train, 
    batch_size=32, 
    epochs=EPOCHS, 
    validation_data=(x_val, y_val),
    verbose=1,
    # callbacks=callbacks_list
)
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