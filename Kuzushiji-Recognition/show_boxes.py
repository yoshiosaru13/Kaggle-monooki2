import glob
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files = glob.glob('datasets/train_images/*.jpg')
img = Image.open(files[0])

df_train = pd.read_csv('datasets/train.csv')

fontsize = 50

font = ImageFont.truetype('datadets/NotoSansCJKjp-Regular.otf', fontsize, encoding='utf-8')
unicode_map = {codepoint: char for codepoint, char in pd.read_csv('datasets/unicode_translation.csv').values}

# This function takes in a filename of an image, and the labels in the string format given in train.csv, and returns an image containing the bounding boxes and characters annotated
def visualize_training_data(image_fn, labels):
    # Convert annotation string to array
    labels = np.array(labels.split(' ')).reshape(-1, 5)
    
    # Read image
    imsource = Image.open(image_fn).convert('RGBA')
    bbox_canvas = Image.new('RGBA', imsource.size)
    char_canvas = Image.new('RGBA', imsource.size)
    bbox_draw = ImageDraw.Draw(bbox_canvas) # Separate canvases for boxes and chars so a box doesn't cut off a character
    char_draw = ImageDraw.Draw(char_canvas)

    for codepoint, x, y, w, h in labels:
        x, y, w, h = int(x), int(y), int(w), int(h)
        char = unicode_map[codepoint] # Convert codepoint to actual unicode character

        # Draw bounding box around character, and unicode character next to it
        bbox_draw.rectangle((x, y, x+w, y+h), fill=(255, 255, 255, 0), outline=(255, 0, 0, 255))
        char_draw.text((x + w + fontsize/4, y + h/2 - fontsize), char, fill=(0, 0, 255, 255), font=font)

    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)
    imsource = imsource.convert("RGB") # Remove alpha for saving in jpg format.
    return np.asarray(imsource)

np.random.seed(1337)

for i in range(10):
    img, labels = df_train.values[np.random.randint(len(df_train))]
    viz = visualize_training_data('datasets/train_images/{}.jpg'.format(img), labels)
    
    plt.figure(figsize=(15, 15))
    plt.title(img)
    plt.imshow(viz, interpolation='lanczos')
    plt.show()