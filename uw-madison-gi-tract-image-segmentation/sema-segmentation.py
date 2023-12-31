# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ezlQIj8RelJepp420W0tGlwQhxybuCM-
"""

import glob
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

file_paths = glob.glob('drive/MyDrive/segmentation/case2/')

df = pd.read_csv('drive/MyDrive/segmentation/train.csv')
df = df[df['segmentation'].notnull()]

df['case'] = df.apply(lambda x: x['id'].split('_')[0][4:], axis=1)
df['day'] = df.apply(lambda x: x['id'].split('_')[1][3:], axis=1)
df['slice'] = df.apply(lambda x: x['id'].split('_')[3], axis=1)

df = df.drop('id', axis=1)


root_path = 'drive/MyDrive/segmentation/'

df['path'] = df.apply(lambda x: root_path + 'case' + x['case'] +
                      '/case' + x['case'] + '_day' + x['day'] +
                      '/scans/' + 'slice_' + x['slice'] + '*.png', axis=1)
#df['path'] = df.apply(lambda x: root_path + 'case' + x['case'] + '/' +
                     # 'case' + x['case'] + '_day' + x['day'] + '/' +
                    #  'scans/' + 'slice_' + x[slice] + '*.png', axis=1)
print(df.iloc[0, -1])
print(df)

file_name = 'drive/MyDrive/segmentation/case2/case2_day1/scans/slice_0001*.png'

image = Image.open(glob.glob(file_name)[0])
image.show()

classes = df['class'].unique()
dictionary = {c: i for i, c in enumerate(classes)}

df['class'] = df['class'].map(dictionary)

df = df.drop('case', axis=1)
df = df.drop('day', axis=1)
df = df.drop('slice', axis=1)

print(df)

X = []
X.append(df['path'].values)
X = np.array(X).squeeze()
print(X.shape)
X = np.unique(X)
print(X.shape)
print(X[10])

index = np.where(X == df.iloc[2, 2])[0]
print(index)
print(df.iloc[2, 2])
print(X[index])

def decoder(seg):
    s = np.asarray(seg.split(), dtype=int)
    starts = s[0::2] - 1
    length = s[1::2]
    ends = starts + length

    img = np.zeros(266*266, dtype=np.uint8)

    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1

    img = img.reshape(266, 266)

    return img

seg = df.loc[194, 'segmentation']

seg_img = decoder(seg)

seg_img *= 255
plt.imshow(seg_img)
plt.show()
print(seg_img.shape)

true_img_path = df.loc[194, 'path']
true_img = Image.open(glob.glob(true_img_path)[0])
true_img = np.array(true_img)

plt.imshow(true_img)
plt.show()

print(seg_img.shape)
print(true_img.shape)

zeros = np.zeros((X.shape[0], 266, 266, 3))
i = 0
for row in df.itertuples(index=False):
    seg_img = decoder(row[1])
    idx = np.where(X == row[2])
    zeros[idx, :, :, row[0]] = seg_img

def show_img(mask, true, class_name):

    mask_bool = np.zeros((266, 266), dtype=bool)
    mask_bool[mask==0] = True

    rgba_mask = np.zeros((266, 266, 4))
    rgba_mask[:, :, class_name] = mask
    rgba_mask[:, :, 3] = np.where(mask_bool, 0, 255)

    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(true, cmap='bone')
    axes[1].imshow(true, cmap='bone')
    axes[1].imshow(rgba_mask)
    plt.show()

show_img(seg_img, true_img, 0)

