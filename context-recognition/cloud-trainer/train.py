import keras
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
from keras.utils import np_utils
from tensorflow.python.lib.io import file_io

import h5py
import argparse
import numpy as np
import tensorflow as tf
import sys
from time import time
from sklearn.metrics import confusion_matrix

def train_model(data_file, job_dir, cnn_model, first_training_epochs, second_training_epochs, data_bucket, **args):
    print('Tensorflow version: ' + tf.__version__)
    print('Keras version: ' + keras.__version__)
    print ('Data bucket: ' + data_bucket)
    print ('Data file: ' + data_file)
    print ('Job directory: ' + job_dir)
    print ('First training epochs: ' + str(first_training_epochs))
    print ('Second training epochs: ' + str(second_training_epochs))

    img_height = 224
    img_width = 224
    batch_size = 32

    train_set_key = 'train_set_'
    valid_set_key = 'valid_set_'
    classes_key = 'list_classes'

    if K.image_data_format() == 'channels_first':
      input_shape = (3, img_width, img_height)
    else:
      input_shape = (img_width, img_height, 3)

    print ('Trying to read file: ' + data_bucket + data_file)
    with file_io.FileIO(data_bucket + data_file, mode='rb') as input_f:
        print ('Opened file')
        with file_io.FileIO(data_file, mode='wb+') as output_f:
                output_f.write(input_f.read())
                print ('Written file to: ' + data_file)

    print ('Copied ' + data_file + ' locally')

    # Read h5 dataset file
    dataset = h5py.File(data_file, 'r')
    list_classes = dataset[classes_key]
    classes = list_classes.shape[0]
    train_set_x = dataset[train_set_key + 'x']
    valid_set_x = dataset[valid_set_key + 'x']

    print ('Converted train/validation set to categorical')
    train_set_y = np_utils.to_categorical(dataset[train_set_key + 'y'], classes)
    valid_set_y = np_utils.to_categorical(dataset[valid_set_key + 'y'], classes)

    print ('Read datasets from h5 file')

    # Create data generator
    datagen = ImageDataGenerator(rescale = 1./255)

    # Create the base pre-trained model
    if cnn_model in 'inceptionv3':
        base_model = InceptionV3(weights ='imagenet', include_top = False, input_shape = input_shape)
        prefix = 'inceptionv3'
        print ('Base model: InceptionV3')
    elif cnn_model in 'densenet121':
        base_model = DenseNet121(weights = 'imagenet', include_top = False, input_shape = input_shape, input_tensor = None)
        prefix = 'densenet121'
        print ('Base model: DenseNet121')
    else:
        print ('Error: ' + model + ' is not a known model')
        sys.exit()

    print ('Created base model with weights pre-trained on ImageNet')

    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Let's add a fully-connected layer
    x = Dense(1024, activation = 'relu')(x)

    print ('Added GlobalAvgPooling2D + Dense')

    # And a logistic layer
    predictions = Dense(classes, activation = 'softmax')(x)

    print ('Added Dense classification layer')

    # This is the model we will train
    model = Model(inputs = base_model.input, outputs = predictions)

    print ('Added extra layers for classification')

    # First: train only the extra top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    print ('Starting fit_generator on model')

    # Fits the model on batches
    model.fit_generator(datagen.flow(train_set_x, train_set_y, batch_size = batch_size),
                        steps_per_epoch = len(train_set_x) / batch_size,
                        epochs = first_training_epochs,
                        validation_data = datagen.flow(valid_set_x, valid_set_y, batch_size = batch_size),
                        validation_steps = len(valid_set_x) / batch_size,
                        callbacks = [])

    print ('Finished training only the extra layers')

    # Freeze only subset of base model's layers
    if cnn_model in 'inceptionv3':
        layers_not_trainable = 249
    elif cnn_model in 'densenet121':
        layers_not_trainable = 120

    for layer in model.layers[:layers_not_trainable]:
        layer.trainable = False
    for layer in model.layers[layers_not_trainable:]:
        layer.trainable = True

    # Recompile model using SGD with a low learning rate
    model.compile(optimizer = SGD(lr = 0.0001, momentum = 0.9), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    print ('Starting fit_generator on model')

    # Fits the model on batches
    model.fit_generator(datagen.flow(train_set_x, train_set_y, batch_size = batch_size),
                        steps_per_epoch = len(train_set_x) / batch_size,
                        epochs = second_training_epochs,
                        validation_data = datagen.flow(valid_set_x, valid_set_y, batch_size = batch_size),
                        validation_steps = len(valid_set_x) / batch_size,
                        callbacks = [])

    # Save the model locally
    print ('Saved model to :' + prefix + '-model.h5')
    model.save(prefix + '-model.h5')

    # Save the model to the Cloud Storage bucket's jobs directory
    with file_io.FileIO(prefix + '-model.h5', mode='rb') as input_f:
        with file_io.FileIO(job_dir + '/' + prefix + '-model.h5', mode='wb+') as output_f:
            output_f.write(input_f.read())

if __name__ == '__main__':

    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--data_bucket',
      help='Cloud Storage bucket or local path to training data file',
      required=True)

    parser.add_argument(
      '--data_file',
      help='Training data file name',
      required=True)

    parser.add_argument(
      '--job-dir',
      help='Cloud Storage bucket to export the model and store temp files',
      required=True)

    parser.add_argument(
      '--cnn_model',
      help='Model to train (InceptionV3/DenseNet121/ResNet50)',
      default='dense')

    parser.add_argument(
      '--first_training_epochs',
      help='Number of epochs for training only the extra layers',
      type=int,
      default=3)

    parser.add_argument(
      '--second_training_epochs',
      help='Number of epochs for training the extra layers plus some base layers',
      type=int,
      default=8)

    args = parser.parse_args()
    arguments = args.__dict__

    print(arguments)

    train_model(**arguments)