'''
Author: Focsaneanu Andrei-Ciprian

Fine tunning of InceptionV3/Resnet50/DenseNet121 models pre-trained on ImageNet:
- add pooling/dense/dense layers for new categories
- make only extra layers trainable
- train for a number of epochs
- make a smaller subset of base model untrainable
- train for a number of epochs
'''

from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D, Flatten
from keras.models import Model, model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

import numpy as np
import tensorflow as tf
import sys
from time import time
from sklearn.metrics import confusion_matrix

# Define flags for input arguments
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'inceptionv3', 'Base model')
flags.DEFINE_string('data_directory', 'subset27x2000', 'Data directory')
flags.DEFINE_integer('first_training_epochs', 5, 'First epochs')
flags.DEFINE_integer('second_training_epochs', 20, 'Second epochs')
flags.DEFINE_integer('batch_size', 25, 'Batch size')
flags.DEFINE_integer('img_width', 224, 'Image width')
flags.DEFINE_integer('img_height', 224, 'Image height')

# Directories used
train_data_dir = FLAGS.data_directory + '\\train'
validation_data_dir = FLAGS.data_directory + '\\test'
test_data_dir = FLAGS.data_directory + '\\test'
print (train_data_dir)

if K.image_data_format() == 'channels_first':
    input_shape = (3, FLAGS.img_width, FLAGS.img_height)
else:
    input_shape = (FLAGS.img_width, FLAGS.img_height, 3)

# Create data generators
train_datagen = ImageDataGenerator(rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (FLAGS.img_width, FLAGS.img_height),
    batch_size = FLAGS.batch_size,
    shuffle = True)

classes = len(train_generator.class_indices)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(FLAGS.img_width, FLAGS.img_height),
    batch_size = FLAGS.batch_size)

# Create the base pre-trained model
if FLAGS.model in 'inceptionv3':
    base_model = InceptionV3(weights ='imagenet', include_top = False, input_shape = input_shape)
    prefix = 'inceptionv3'
    print ('Base model: InceptionV3')
elif FLAGS.model in 'resnet50':
    base_model = ResNet50(weights = 'imagenet', include_top = False, input_shape = input_shape)
    prefix = 'resnet50'
    print ('Base model: ResNet50')
elif FLAGS.model in 'densenet121':
    base_model = DenseNet121(weights = 'imagenet', include_top = False, input_shape = input_shape, input_tensor = None)
    prefix = 'densenet121'
    print ('Base model: DenseNet121')
else:
    print ('Error: ' + FLAGS.model + ' is not a known model')
    sys.exit()

# Add a global spatial average pooling layer
x = base_model.output
if FLAGS.model in 'resnet50':
    x = Flatten(name="flatten")(x)
else:
    x = GlobalAveragePooling2D()(x)
    # Let's add a fully-connected layer
    x = Dense(1024, activation = 'relu')(x)
# And a logistic layer
predictions = Dense(classes, activation = 'softmax')(x)

# This is the model we will train
model = Model(inputs = base_model.input, outputs = predictions)

# First: train only the extra top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Define callbacks
tensorboard = TensorBoard(log_dir='logs/' + prefix + "-{}".format(time()))

# Train the model on the new data for a few epochs
model.fit_generator(
    train_generator,
    epochs = FLAGS.first_training_epochs,
    validation_data = validation_generator,
    callbacks = [tensorboard])

print ('Finished training only the extra layers')

# Freeze only subset of base model's layers
if FLAGS.model in 'inceptionv3':
    layers_not_trainable = 249
elif FLAGS.model in 'resnet50':
    layers_not_trainable = 120
elif FLAGS.model in 'densenet121':
    layers_not_trainable = 120

for layer in model.layers[:layers_not_trainable]:
    layer.trainable = False
for layer in model.layers[layers_not_trainable:]:
    layer.trainable = True

# Recompile model using SGD with a low learning rate
model.compile(optimizer = SGD(lr = 0.0001, momentum = 0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint('model-' + prefix + '-{epoch:03d}.h5', verbose = 1, monitor = 'val_loss', save_best_only = True, mode = 'auto')
es = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3, verbose = 0, mode = 'auto')

# Train model
model.fit_generator(
    train_generator,
    epochs = FLAGS.second_training_epochs,
    validation_data = validation_generator,
    callbacks = [tensorboard, checkpoint, es])

# Generate confusion matrix
print ('Generating confusion matrix..')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size = (FLAGS.img_width, FLAGS.img_height),
    batch_size = 1,
    shuffle = False)
test_elements = test_generator.__len__()
probabilities = model.predict_generator(test_generator, test_elements)
y_pred = np.argmax(probabilities, axis = 1)
y_true = np.array([], dtype = int)
for class_index in range(0, classes):
    for index in range(0, int(test_elements/classes)):
        y_true = np.append(y_true, class_index)
mat = np.matrix(confusion_matrix(y_true, y_pred))
with open('confusion.txt','wb') as f:
    for line in mat:
        np.savetxt(f, line, fmt='%d')
print ('Wrote confusion matrix to: confusion.txt')

# Save model weights
model_json = model.to_json()
model_filename = prefix + '-model'
with open(model_filename, "w") as json_file:
    json_file.write(model_json)
print ("Saved model to disk")