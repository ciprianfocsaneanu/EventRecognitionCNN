import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys

with open('confusions/confusion-inceptionv3-19042018.txt','rb') as f:
    m = np.loadtxt(f)

# Constants used
img_width, img_height = 224, 224
test_data_dir = 'places205-subset27\\test'

validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = validation_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=25)

classes = len(validation_generator.class_indices)
categories = classes * ['']

for key, value in validation_generator.class_indices.items():
    categories[value] = key

for x in range(0, 27):
    for y in range(0, 27):
        m[x,y] = int(float("{0:.2f}".format(m[x,y]/399 *100)))

m = m.astype(int)
df_cm = pd.DataFrame(m, categories, categories)
sn.set(font_scale=0.9)#for label size
sn.heatmap(df_cm, cmap="BuPu", annot=True, annot_kws={"size": 14})# font size

plt.show()