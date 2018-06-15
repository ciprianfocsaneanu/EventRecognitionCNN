import keras
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.densenet import DenseNet121
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.layers import Dense, GlobalAveragePooling2D, Flatten, concatenate, Input
from keras.models import Model, model_from_json, load_model
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

def train_model(data_file, job_dir, context_model_file, data_bucket, training_epochs, **args):
    print('Tensorflow version: ' + tf.__version__)
    print('Keras version: ' + keras.__version__)
    print ('Data bucket: ' + data_bucket)
    print ('Data file: ' + data_file)
    print ('Context model file: ' + context_model_file)
    print ('Job directory: ' + job_dir)
    print ('Training epochs: ' + str(training_epochs))

    ## Prepare training data
    img_height = 224
    img_width = 224
    batch_size = 16

    if K.image_data_format() == 'channels_first':
      input_shape = (3, img_width, img_height)
    else:
      input_shape = (img_width, img_height, 3)

    train_set_key = 'train_set_'
    valid_set_key = 'valid_set_'
    classes_key = 'list_classes'

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

    ## Give same input to both models
    model_input = Input(shape = input_shape)

    print ('Trying to read file: ' + data_bucket + context_model_file)
    with file_io.FileIO(data_bucket + context_model_file, mode='rb') as input_f:
        print ('Opened file')
        with file_io.FileIO(context_model_file, mode='wb+') as output_f:
          output_f.write(input_f.read())
          print ('Written file to: ' + context_model_file)

    print ('Copied ' + context_model_file + ' locally')
    ## Load context recognition model pre-trained on 27 categories of Places205
    loaded_context_model = load_model(context_model_file)

    # Construct new context CNN model with pre-trained weights loaded but connected to input tensor
    context_cnn = DenseNet121(weights = None, include_top = False, input_tensor = model_input)
    x = GlobalAveragePooling2D()(context_cnn.output)
    x = Dense(1024, activation = 'relu')(x)
    context_predictions = Dense(27, activation = 'softmax')(x)
    context_model = Model(inputs = model_input, outputs = context_predictions)
    for new_layer, layer in zip(context_model.layers[1:], loaded_context_model.layers[1:]):
        new_layer.set_weights(layer.get_weights())

    ## Load object recognition model pre-trained on ImageNet
    object_model = ResNet50(include_top=True, weights='imagenet', input_tensor = model_input, pooling=None, classes=1000)
    # Hack to avoid https://github.com/keras-team/keras/issues/3974
    for i in range(1,len(object_model.layers)):
        object_model.layers[i].name = object_model.layers[i].name + '1'

    ##  Merge/concatenate outputs of the two sub-models
    context_output = context_model.output
    object_output = object_model.output
    y = concatenate([context_output, object_output])

    ##  Add extra layer(s) for classification of activities
    y = Dense(1024, activation = 'relu')(y)
    predictions = Dense(classes, activation = 'softmax')(y)

    ## Set C-CNN layers to be not trainable
    for layer in context_model.layers:
        layer.trainable = False

    ## Compile model
    model = Model(inputs = model_input, outputs = predictions)
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

    model_filename = 'activity-recognition-model.h5'
    checkpoint = ModelCheckpoint(model_filename, verbose = 1, monitor = 'val_acc', save_best_only = True, mode = 'auto')

    print ('Starting fit_generator on model')
    # Fits the model on batches
    model.fit_generator(datagen.flow(train_set_x, train_set_y, batch_size = batch_size),
                        steps_per_epoch = len(train_set_x) / batch_size,
                        epochs = training_epochs,
                        validation_data = datagen.flow(valid_set_x, valid_set_y, batch_size = batch_size),
                        validation_steps = len(valid_set_x) / batch_size,
                        callbacks = [checkpoint])
    print ('Finished training only the extra layers')

    # # Save the model locally
    # model.save(model_filename)
    # print ('Saved model to :' + model_filename)

    # Save the model to the Cloud Storage bucket's jobs directory
    with file_io.FileIO(model_filename, mode='rb') as input_f:
        with file_io.FileIO(job_dir + '/' + model_filename, mode='wb+') as output_f:
            output_f.write(input_f.read())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--data_bucket',
      help='Cloud Storage bucket where dataset is located',
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
      '--context_model_file',
      help='Filename of context recognition model',
      required=True)

    parser.add_argument(
      '--training_epochs',
      help='Number of epochs for training only the extra layers',
      type=int,
      default=20)

    args = parser.parse_args()
    arguments = args.__dict__

    train_model(**arguments)