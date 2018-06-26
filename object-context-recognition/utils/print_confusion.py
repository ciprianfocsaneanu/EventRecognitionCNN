import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys

with open('./confusion-oc-cnn-1528961158.067217.txt','rb') as f:
    m = np.loadtxt(f)

# Constants used
img_width, img_height = 224, 224
test_data_dir = './newWIDER\\test'

validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = validation_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=25)

classes = len(validation_generator.class_indices)
categories = classes * ['']

for key, value in validation_generator.class_indices.items():
    categories[value] = key

sum = classes * [0]

for x in range(0, classes):
    for y in range(0, classes):
        m[x,y] = int(float("{0:.2f}".format(m[x,y])))
        sum[x] += m[x,y]

for x in range(0, classes):
    own_ratio = float(m[x,x]) / sum[x]
    if (own_ratio > 0.15):
        print ('Category ' + str(categories[x]) + ' has score ' + str(own_ratio))
        max_conf = 0
        max_y = 0
        for y in range(0, classes):
            if x != y and float(m[x,y])/sum[x] > max_y:
                max_conf = float(m[x,y])/sum[x]
                max_y = y
        print ('gets most confused with ' + str(categories[max_y]) + ' with score ' + str(max_conf))

for x in range(0, classes):
    for y in range(0, classes):
        m[x,y] = float(m[x,y])/sum[x] *100

m = m.astype(int)
df_cm = pd.DataFrame(m, categories, categories)
sn.set(font_scale=1.3) # for label size
sn.heatmap(df_cm, cmap="BuPu", annot=True, annot_kws={"size": 14}) # font size

plt.show()