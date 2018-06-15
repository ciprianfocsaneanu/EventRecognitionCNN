from keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Constants used
img_width, img_height = 224, 224
test_data_dir = 'subset28x2000\\test'

validation_datagen = ImageDataGenerator(rescale = 1./255)
validation_generator = validation_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=25)

print (validation_generator.class_indices)

with open('confusion.txt','rb') as f:
    m = np.loadtxt(f)

col = 28
frecv = col * [0]

for i in range(0, col):
    target = m[i][i]
    if target < 250:
        for key, value in validation_generator.class_indices.items():
            if value == i:
                print (str(key) + ' has ' + str(target) + ' / 399')
        for j in range(0, col):
            if m[i][j] >= 30 and i != j:
                frecv[j] = frecv[j] + 1
                for key, value in validation_generator.class_indices.items():
                    if value == j:
                        print ('is confused with ' + str(key) + ' with ' + str(m[i][j]) + ' / 399')

# print('Most confusing categories (as frequency):')
# for i in range(0, col):
#     for key, value in validation_generator.class_indices.items():
#         if value == i:
#             print(str(key) + ' is confused for ' + str(frecv[i]) + ' times!')