import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Reshape, Multiply, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow.keras.utils as utils
from sklearn.model_selection import train_test_split
import json as simplejson
from tensorflow.keras.models import Model
# Define the input shape of your model
input_shape = (224, 224, 3)
channels = 3

def se_block(input_tensor, ratio=16):
    # squeeze operation
    channels = input_tensor.shape[-1]
    x = GlobalAveragePooling2D()(input_tensor)
    x = Reshape((1, 1, channels))(x)
    x = Dense(channels // ratio, activation='relu', use_bias=False)(x)
    x = Dense(channels, activation='sigmoid', use_bias=False)(x)
    x = Multiply()([input_tensor, x])
    return x

# Load the ResNet50 model with pre-trained ImageNet weights
resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the weights of the pre-trained model
for layer in resnet.layers:
    layer.trainable = True

# Add the SE block after the last convolutional block
x = resnet.output
x = se_block(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(256, activation='relu')(x)
outputs = Dense(channels, activation='softmax')(x)

# Create the final model
model = Model(inputs=resnet.input, outputs=outputs)

# Create a new model instance with the top layer
# model = tf.keras.Sequential([
#     base_model,
#     # tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(256, activation='relu'),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(3, activation='softmax')
# ])

print(model.summary())

lr = 1e-4
optimizer = Adam(learning_rate=lr)

METRICS = [
      "accuracy",
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

# Compilar modelo
model.compile(
    loss='categorical_crossentropy',
    optimizer=optimizer,
    metrics=METRICS,
)

# Fit model (storing  weights) -------------------------------------------
filepath="./weights/weights_resnet.hdf5"
checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, 
                             monitor='val_accuracy', 
                             verbose=1, 
                             save_best_only=True, 
                             mode='max')

lr_reduce   = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=1e-5, patience=3, verbose=0)
early       = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, mode='max')
checkpoint  = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint, lr_reduce, early]

print('IMPORT DATA -----------------------------')

x_all = np.load('../../dataset/x_all_v2.npy')
y_all = np.load('../../dataset/y_all_v2.npy')

y_all = utils.to_categorical(y_all, num_classes=3)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Divida o conjunto de dados em conjunto de treinamento e teste
x_train_val, x_test, y_train_val, y_test = train_test_split(x_all, y_all, stratify=y_all, test_size=test_ratio, random_state=123)

# Divida o conjunto de treinamento em conjunto de treinamento e validação
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, stratify=y_train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=123)


print('TRAINING MODEL -----------------------------')

x_train = np.squeeze(x_train)
x_val = np.squeeze(x_val)
x_test = np.squeeze(x_test)

history = model.fit(x_train, y_train, 
          validation_data=(x_val, y_val),
          batch_size=32, 
          epochs=100, 
          verbose=1,
          callbacks=callbacks_list)

### storing Model in JSON --------------------------------------------------

model_json = model.to_json()

with open("./model/model_resnet.json", "w") as json_file:
    json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))


### evaluate model ---------------------------------------------------------

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 