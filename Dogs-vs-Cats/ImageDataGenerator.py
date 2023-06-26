from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import tensorflow as tf
import keras
from keras import layers, Sequential, Input
from keras.applications.resnet import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import pickle
from sklearn.model_selection import train_test_split


EPOCHS=2



train_files = glob.glob('datasets/train/*.jpg')
train_files = [file.replace('datasets/train/', '') for file in train_files]


X_train = []
y_train = []
img_width = 32
img_height = 32
n_classes = 2

for file in train_files:
    if 'cat' in file:
        y_train.append('0')
    else:
        y_train.append('1')
        
        
df = pd.DataFrame({'filename':train_files, 'label':y_train})

print(df)

train_df, valid_df = train_test_split(df, test_size = 0.2, stratify = df['label'], random_state = 123)
print(train_df.shape)
print(valid_df.shape)

train_datagen = ImageDataGenerator(rotation_range = 10, zoom_range = 0.1, horizontal_flip = True, fill_mode = 'nearest', 
                                   width_shift_range = 0.1, height_shift_range = 0.1, preprocessing_function = preprocess_input)

#flow_from_dataframe() method will accept dataframe with filenames as x_column and labels as y_column to generate mini-batches
train_gen = train_datagen.flow_from_dataframe(train_df, directory = 'datasets/train', x_col = 'filename', y_col = 'label', target_size = (64,64),
                                              batch_size = 64, class_mode='binary')

#we do not augment validation data.
valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

valid_gen = valid_datagen.flow_from_dataframe(valid_df, directory = 'datasets/train', x_col = 'filename', y_col = 'label', target_size = (64,64),
                                              batch_size = 64, class_mode='binary')

model = Sequential()
model.add(ResNet50(include_top = False, pooling = 'max', weights = 'imagenet'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1, activation = 'sigmoid'))

model.layers[0].trainable = False 

model.compile(optimizer = 'adam', metrics = ['accuracy'], loss = 'binary_crossentropy')
model.summary()

#model.fit_generator(train_gen, epochs = 10, validation_data = valid_gen)

loss = pd.DataFrame(model.history.history)
loss[['loss', 'val_loss']].plot()
loss[['accuracy', 'val_accuracy']].plot()