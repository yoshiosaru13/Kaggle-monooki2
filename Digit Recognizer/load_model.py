from tkinter import Image
from CNN_model import X_train, X_test, y_train
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input, Activation
from keras.layers import Conv2D, Flatten, Reshape, LeakyReLU, MaxPooling2D
from keras.models import Model
import tensorflow as tf

new_model = tf.keras.models.load_model('saved_model/my_model.h5')
new_model.fit(X_train, y_train, epochs=20, validation_split=0.1, batch_size=64)

predictions = new_model.predict_step(X_test)
pred_class = np.argmax(predictions, axis=1)

ImageId = np.arange(1, len(pred_class)+1)

submittion = pd.DataFrame({"ImageId": ImageId, "Label": pred_class})
submittion.to_csv('CNN_epoch20.csv', index=False)
