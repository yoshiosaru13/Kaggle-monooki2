from cgi import test
import glob
import pandas as pd

from sklearn.model_selection import train_test_split
import keras
from keras.preprocessing.image import ImageDataGenerator

train_files = glob.glob('datasets/train/*.jpg')
train_files = [file.replace('datasets/train/', '') for file in train_files]

y_train = []

for file in train_files:
    if 'cat' in file:
        y_train.append('0')
    else:
        y_train.append('1')
        
df = pd.DataFrame({'filename':train_files, 'label':y_train})

train_df, val_df = train_test_split(df, test_size=0.2, stratify = df['label'], random_state=66)

train_datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1)