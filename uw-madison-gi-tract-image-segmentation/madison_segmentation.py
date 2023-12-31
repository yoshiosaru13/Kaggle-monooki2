# -*- coding: utf-8 -*-
"""Madison-Segmentation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1afbFgFo5eLLxPkLTk75a54oiuxENHhf4
"""

import glob
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

files_path = glob.glob('drive/MyDrive/segmentation/case2/case2_day1/scans/*.png')
files_path.sort()
img_path = files_path[58]

df = pd.read_csv('drive/MyDrive/segmentation/train.csv')
df_img = df.iloc[110478, :]
img_seg = df_img['segmentation']

print(img_path)
print(df_img)
print(img_seg)

s = np.asarray(img_seg.split(), dtype=int)
starts = s[0::2] - 1
length = s[1::2]
ends = starts + length

img = np.zeros(266*266, dtype=np.uint8)

for lo, hi in zip(starts, ends):
    img[lo:hi] = 1

img = img.reshape(266, 266)

shape = (266, 266, 3)
mask = np.zeros(shape, dtype=np.uint8)
mask[..., 0] = img
mask *= 255

image = Image.open(img_path)
image = np.asarray(image).astype('uint8')
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
image = clahe.apply(image)

fig, axes = plt.subplots(1, 3)
axes[0].imshow(mask, alpha=0.5)
axes[1].imshow(image, cmap='bone')
axes[2].imshow(image, cmap='bone')
axes[2].imshow(mask, alpha=0.5)

plt.show()

print(mask.shape)
array_bool = np.all(mask == [0, 0, 0], axis=2)
print(array_bool.shape)
print(array_bool)
c_mask = np.copy(mask)
c_mask[:, :, 0] = np.where(array_bool, 255, c_mask[:, :, 0])
c_mask[:, :, 1] = np.where(array_bool, 255, c_mask[:, :, 1])
c_mask[:, :, 2] = np.where(array_bool, 255, c_mask[:, :, 2])

rgba_mask = np.zeros((266, 266, 4), dtype=np.uint8)
rgba_mask[:, :, :3] = c_mask

rgba_mask[:, :, 3] = np.where(array_bool, 0, 255)

print(c_mask)
plt.imshow(rgba_mask)
plt.show()

all_zeros = np.zeros([266, 266, 3], dtype=int)
all_one = all_zeros + 1
all_255 = all_one * 255
print(all_zeros[0][0].shape)
plt.imshow(all_255)
plt.show()
print(all_255)

image = Image.open(img_path)
image = np.asarray(image).astype('uint8')
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
image = clahe.apply(image)

fig, axes = plt.subplots(1, 3)
axes[0].imshow(rgba_mask)
axes[1].imshow(image, cmap='bone')
axes[2].imshow(image, cmap='bone')
axes[2].imshow(rgba_mask)

plt.show()