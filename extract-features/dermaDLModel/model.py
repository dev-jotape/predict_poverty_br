import os
import pickle
from glob import glob
from random import Random
import numpy as np
import pandas as pd
import sklearn.metrics as skl_metrics
import tensorflow.keras.utils as utils
from sklearn.model_selection import train_test_split
import tensorflow as tf


print('STEP 1: MODEL DEFINITION')

depth = lambda x: x.shape[-1]
if tf.keras.backend.backend() != 'tensorflow':
    depth = lambda x: x.shape.dims[-1]

def cn_block(x, filters, kernel, stride, name='cn', ridge=0.0005):
    ''' convolutional block = conv2d + batchnorm + relu '''
    r = tf.keras.regularizers.l2(ridge)
    x = tf.keras.layers.Conv2D(filters, kernel, strides=stride, padding='same',
            use_bias=False, kernel_regularizer=r, name=f'{name}_conv')(x)
    x = tf.keras.layers.BatchNormalization(name=f'{name}_bn')(x)
    return tf.keras.layers.Activation('relu')(x)

def fc_block(x, out, name='fc', ridge=0.0005):
    ''' fully-connected (dense) block + dropout '''
    r = tf.keras.regularizers.l2(ridge)
    x = tf.keras.layers.Dense(out, use_bias=False, kernel_regularizer=r, name=name)(x)
    return tf.keras.layers.Dropout(0.5)(x)

def se_block(i, out, name='se', reduction=2):
    ''' squeeze-excite attention block '''
    x = tf.keras.layers.GlobalAvgPool2D()(i)
    x = fc_block(x, out//reduction, name=f'{name}_fc1')
    x = tf.keras.layers.Activation('relu')(x)
    x = fc_block(x, out, name=f'{name}_fc2')
    x = tf.keras.layers.Activation('sigmoid')(x)
    x = tf.keras.layers.Reshape((1,1,out))(x)
    return tf.keras.layers.multiply([i,x])

def id_block(i, out, num, cardinality1=3, cardinality2=8, split_out=8):
    ''' aggregated residual + se block '''
    ch = depth(i)
    if ch == out:
        s, p = 1, i
    else:
        assert(out == 2*ch)
        s = 2
        p = tf.keras.layers.AvgPool2D()(i)
        p = tf.keras.layers.concatenate([p,p])
    xr = []
    for j in range(cardinality1):
        xs = []
        for k in range(cardinality2):
            x = cn_block(i, split_out, (1,1), 1, f'res_split{num}{j:x}{k:x}1')
            x = cn_block(x, split_out, (3,3), s, f'res_split{num}{j:x}{k:x}2')
            xs.append(x)
        x = tf.keras.layers.concatenate(xs)
        x = cn_block(x, out, (1,1), 1, name=f'res_tr{num}{j:x}')
        x = se_block(x, out, f'res_se{num}{j:x}')
        x = tf.keras.layers.add([p,x])
        x = tf.keras.layers.Activation('relu')(x)
        xr.append(x)
    x = tf.keras.layers.concatenate(xr)
    x = cn_block(x, out, (1,1), 1, name=f'res_join{num}')
    return x

def DermaDL(input_shape=(224,224,3), outputs=3, activation='softmax', d_out=128, **opts):
    ''' our network with multi-class output '''
    i = tf.keras.Input(input_shape)
    x = cn_block(i, d_out//8, (3,3), 1, 'init')
    x = id_block(x, d_out//8, 0, **opts)
    x = id_block(x, d_out//4, 1, **opts)
    x = id_block(x, d_out//2, 2, **opts)
    x = id_block(x, d_out//1, 3, **opts)
    x = tf.keras.layers.GlobalAvgPool2D(name='last_pool')(x)
    x = fc_block(x, d_out, name='last_fc')
    x = tf.keras.layers.Dense(outputs, activation=activation, name='logits')(x)
    return tf.keras.Model(i,x)

def get_df(path, balance=True, shuffle=True, samples=None, seed=999991):
    ''' dataset loading helper '''
    rng = Random(seed)
    data = []
    for classdir in glob(os.path.join(path,'*')):
        classname = classdir.split(os.path.sep)[-1]
        files = [] 
        for filename in glob(os.path.join(classdir,'*')):
            files.append([filename, classname])
        if shuffle:
            rng.shuffle(files)
        if samples:
            files = files[:samples]
        data.append(files)
    df = []
    if balance:
        n = max([len(x) for x in data])
        for l in [rng.choices(x,k=n) for x in data]:
            df.extend(l)
    else:
        for l in data:
            df.extend(l)
    if shuffle:
        rng.shuffle(df)
    return pd.DataFrame(df, columns=['filename','class'])

print('STEP 2: IMPORT DATASET')

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

x_train = np.squeeze(x_train)
x_val = np.squeeze(x_val)
x_test = np.squeeze(x_test)

print('STEP 3: TRAINING')

loss, f = 'categorical_crossentropy', 'softmax'
bsize=32
epochs=100

model = DermaDL()

optim = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(optim, loss, metrics=['categorical_accuracy'])
print('Model size:',model.count_params())
with open('./model.json','w') as f:
    f.write(model.to_json())
h = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=bsize,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                   tf.keras.callbacks.ModelCheckpoint('./weights.h5',
                       save_best_only=True, save_weights_only=True)])
with open('./history.pkl','wb') as f:
    pickle.dump(h.history, f)
model.save_weights('./weights.h5')
print('Saved')

print('STEP 4: EVALUATION')

# with open('./model.json','r') as f:
#     model = tf.keras.models.model_from_json(f.read())
# model.load_weights('./weights.h5')

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 
