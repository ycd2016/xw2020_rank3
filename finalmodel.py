import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

train = pd.read_csv('train_final.csv')
test = pd.read_csv('test_final.csv')
sub = pd.read_csv('example.csv')
y = train.groupby('fragment_id')['behavior_id'].min()

train['mod'] = (train.acc_x ** 2 + train.acc_y ** 2 + train.acc_z ** 2) ** .5
train['modg'] = (train.acc_xg ** 2 + train.acc_yg ** 2 + train.acc_zg ** 2) ** .5
test['mod'] = (test.acc_x ** 2 + test.acc_y ** 2 + test.acc_z ** 2) ** .5
test['modg'] = (test.acc_xg ** 2 + test.acc_yg ** 2 + test.acc_zg ** 2) ** .5


def padding(x, len=60):
    while True:
        if x.shape[0] >= len: break
        x = np.concatenate((x, x), axis=0)
    return x[:60]


x = np.zeros((15000, 60, 8, 1))
t = np.zeros((16000, 60, 8, 1))
for i in tqdm(range(15000)):
    tmp = train[train.fragment_id == i][:60]
    x[i,:,:, 0] = padding(tmp.drop(['fragment_id', 'time_point', 'behavior_id'], axis=1), 60)
for i in tqdm(range(16000)):
    tmp = test[test.fragment_id == i][:60]
    t[i,:,:, 0] = padding(tmp.drop(['fragment_id', 'time_point'], axis=1), 60)

mas = tf.convert_to_tensor([1, 2, 3, 4, 3, 5, 2, 5, 1, 1, 2, 3, 5, 4, 4, 6, 7, 8, 9, 0])
mna = tf.convert_to_tensor([0, 1, 0, 0, 1, 2, 2, 0, 1, 2, 0, 2, 1, 1, 2, 3, 4, 5, 6, 7])


def comboAcc(true, pred):
    atrue = K.argmax(true, axis=-1)
    apred = K.argmax(pred, axis=-1)
    maskas = (tf.gather(mas, atrue) == tf.gather(mas, apred)) & (
        tf.gather(mna, atrue) != tf.gather(mna, apred))
    maskna = (tf.gather(mna, atrue) == tf.gather(mna, apred)) & (
        tf.gather(mas, atrue) != tf.gather(mas, apred))
    acc = K.mean(tf.cast(atrue == apred, tf.float32))
    acc += K.mean(tf.cast(maskas, tf.float32)) / 3.
    acc += K.mean(tf.cast(maskna, tf.float32)) / 7.
    return acc


def comboLoss(true, pred):
    atrue = K.argmax(true, axis=-1)
    apred = K.argmax(pred, axis=-1)
    maskas = tf.where((tf.gather(mas, atrue) == tf.gather(mas, apred)) &
                      (tf.gather(mna, atrue) != tf.gather(mna, apred)))
    maskna = tf.where((tf.gather(mna, atrue) == tf.gather(mna, apred)) &
                      (tf.gather(mas, atrue) != tf.gather(mas, apred)))
    loss = categorical_crossentropy(true, pred)
    tf.tensor_scatter_nd_update(loss, maskas, tf.gather(loss, maskas) * 2. / 3.)
    tf.tensor_scatter_nd_update(loss, maskna, tf.gather(loss, maskna) * 6. / 7.)
    return .25 * loss * (1. - K.exp(-loss)) ** 2.


def Mish(x):
    return x * K.tanh(K.softplus(x))


def BMC(x, nb_filter, kernel_size=3, strides=1, padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    x = BatchNormalization(name=bn_name)(x)
    x = Mish(x)
    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, name=conv_name)(x)
    return x


def Net(l=60, cols_len=8):
    input = Input(shape=(l, cols_len, 1), name='input')
    X = Dropout(0.1, noise_shape=(None, 1, cols_len, 1))(input)
    shortcut = Conv2D(1024, kernel_size=(1, 8), padding='valid')(X)
    X = Conv2D(256, kernel_size=(1, 8), padding='valid')(X)
    X = BMC(X, nb_filter=512, kernel_size=(2, 1))
    X = BMC(X, nb_filter=1024, kernel_size=(2, 1))
    X = AdditiveAttention(causal=True)([shortcut, X])
    shortcut = BMC(X, nb_filter=1024, kernel_size=(4, 1))
    X = BMC(X, nb_filter=256, kernel_size=(4, 1))
    X = BMC(X, nb_filter=512, kernel_size=(2, 1))
    X = BMC(X, nb_filter=1024, kernel_size=(2, 1))
    X = AdditiveAttention(causal=True)([shortcut, X])
    X = AdditiveAttention(causal=True)([X, X])
    X = GlobalAveragePooling2D()(X)
    X = Dense(512, activation='relu')(Flatten()(X))
    X = BatchNormalization()(Dropout(0.2)(X))
    X = Dense(20, activation='softmax')(X)
    model = Model([input], X)
    return model


kfold = StratifiedKFold(5, shuffle=True)

proba_t = np.zeros((16000, 20))
for fold, (xx, yy) in enumerate(kfold.split(x, y)):
    y_ = to_categorical(y, num_classes=20)
    model = Net()
    model.compile(loss=comboLoss,
                  optimizer=Adam(amsgrad=True),
                  metrics=[comboAcc])
    plateau = ReduceLROnPlateau(monitor="val_comboAcc",
                                verbose=0,
                                mode='max',
                                factor=0.1,
                                patience=15)
    early_stopping = EarlyStopping(monitor='val_comboAcc',
                                   verbose=0,
                                   mode='max',
                                   patience=30)
    checkpoint = ModelCheckpoint(f'fold{fold}.h5',
                                 monitor='val_comboAcc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)
    model.fit(x[xx], y_[xx],
              epochs=300,
              batch_size=64,
              verbose=1,
              shuffle=True,
              validation_data=(x[yy], y_[yy]),
              callbacks=[plateau, early_stopping, checkpoint])
    model.load_weights(f'fold{fold}.h5')
    proba_t += model.predict(t, verbose=0, batch_size=640) / 5.

behavior_id = np.argmax(proba_t, axis=1)
confidence = np.max(proba_t, axis=1)
for i in tqdm(range(20)):
    cur = behavior_id == i
    mid = np.median(confidence[cur])
    mask = confidence[cur] > mid
    x = np.concatenate((x, t[cur][mask]), axis=0)
    y = np.concatenate((y, behavior_id[cur][mask]), axis=0)

proba_t = np.zeros((16000, 20))
for fold, (xx, yy) in enumerate(kfold.split(x, y)):
    y_ = to_categorical(y, num_classes=20)
    model = Net()
    model.compile(loss=comboLoss,
                  optimizer=Adam(amsgrad=True),
                  metrics=[comboAcc])
    plateau = ReduceLROnPlateau(monitor="val_comboAcc",
                                verbose=0,
                                mode='max',
                                factor=0.1,
                                patience=15)
    early_stopping = EarlyStopping(monitor='val_comboAcc',
                                   verbose=0,
                                   mode='max',
                                   patience=30)
    checkpoint = ModelCheckpoint(f'fold{fold}.h5',
                                 monitor='val_comboAcc',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)
    model.fit(x[xx], y_[xx],
              epochs=300,
              batch_size=64,
              verbose=1,
              shuffle=True,
              validation_data=(x[yy], y_[yy]),
              callbacks=[plateau, early_stopping, checkpoint])
    model.load_weights(f'fold{fold}.h5')
    proba_t += model.predict(t, verbose=0, batch_size=640) / 5.

sub.behavior_id = np.argmax(proba_t, axis=1)
sub.to_csv('submit.csv', index=False)
