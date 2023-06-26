from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import glob
import tensorflow as tf
import keras
from keras import layers, Sequential, Input
from keras.applications.resnet import ResNet50
import pickle
import tensorflow as tf
from keras.optimizers import Adam, SGD

EPOCHS=2



train_files = glob.glob('datasets/train/*.jpg')



X_train = []
y_train = []
img_width = 224
img_height = 224
n_classes = 2


for file in train_files:
    if 'cat' in file:
        y_train.append(0)
    else:
        y_train.append(1)
    
    img = Image.open(file)
    img = img.convert("RGB")
    img = img.resize((img_width, img_height))
    X = np.asarray(img)
    X_train.append(X)
    
X_train = np.array(X_train).astype('float32')
y_train = np.array(y_train)
#y_train = keras.utils.to_categorical(y_train, 2)


data_augmentation = Sequential([
    #layers.Rescaling(1./255), BatchNormalizationがあるのでいらない
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    
])

Resnet = ResNet50(include_top=False, pooling='avg', weights="imagenet", input_shape=(224, 224, 3), classes=2)

for layer in Resnet.layers:
    layer.trainable = False
    
    
model = Sequential()

model.add(Resnet)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=SGD(learning_rate = 0.001, momentum = 0.1),
              metrics=['accuracy'])

history = model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=EPOCHS,
    validation_split=0.2
)

plt.figure(figsize=(12, 6))

plt.plot(history.epoch, history.history['accuracy'], label='train', c='r')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(history.epoch, history.history['val_accuracy'], label='val', c='b', linestyle='-.')


plt.tight_layout()
plt.show()

