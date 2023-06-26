import numpy as np
import pandas as pd

df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

X_train = df.iloc[:, 1:].values
y_train = df.iloc[:, 0].values
X_test = df_test.values

mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train = (X_train - mean_vals)/std_val
X_test = (X_test - mean_vals)/std_val


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input, Activation
from keras.layers import Conv2D, Flatten, Reshape, LeakyReLU, MaxPooling2D
from keras.models import Model
import tensorflow as tf

X_train = tf.compat.v1.reshape(X_train, shape=(-1, 28, 28, 1))
X_test = tf.compat.v1.reshape(X_test, shape=(-1, 28, 28, 1))
y_train = tf.compat.v1.one_hot(indices=y_train, depth=10, dtype=tf.float32)

if __name__ == "__main__":
    model = Sequential()

    model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))       
    
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train, epochs=5, validation_split=0.2, batch_size=64)
    model.save('saved_model/my_model.h5')
