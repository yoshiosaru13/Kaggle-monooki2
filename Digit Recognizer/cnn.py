from calendar import EPOCH
from tkinter import Image
from turtle import title
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import keras
from keras import layers
from keras.models import Sequential
# -------- 前処理 --------

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')


X_train = df_train.iloc[:, 1:].values
y_train = df_train.iloc[:, 0].values
X_test = df_test.values
mean_vals = np.mean(X_train, axis=0)
std_val = np.std(X_train)

X_train = (X_train - mean_vals)/std_val
X_test = (X_test - mean_vals)/std_val
X_train = X_train.reshape([-1, 28, 28, 1])
y_train = tf.one_hot(indices=y_train, depth=10, dtype=tf.float32)

X_test = X_test.reshape([-1, 28, 28, 1])



IMG_SIZE = (28, 28, 1)
EPOCHS = 5




data_augmentation = Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2)
])

model = Sequential([
    layers.Input(shape=IMG_SIZE),
    #layers.Rescaling(1./255),
    #data_augmentation,
    
    layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    
    layers.Dense(10, activation='softmax'),
    
])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=64, validation_split=0.2)

plt.figure(figsize=(10, 10))

plt.subplot(1, 2, 1)
plt.plot(history.epoch, history.history['loss'], label='train', color='red')
plt.plot(history.epoch, history.history['val_loss'], label='val', color='blue', linestyle='-.')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.title('loss')

plt.subplot(1, 2, 2)
plt.plot(history.epoch, history.history['accuracy'], label='train', color='red')
plt.plot(history.epoch, history.history['val_accuracy'], label='val', color='blue', linestyle='-.')
plt.legend(loc='best')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('accuracy')

plt.show()

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
print(y_pred)

ImageId = np.arange(1, 28001)

submitton = pd.DataFrame({'ImageId': ImageId, 'Label': y_pred})
submitton.to_csv('submittion.csv', index=None)




