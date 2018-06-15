'''
Author: Focsaneanu Andrei-Ciprian
June 2018


'''
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

import argparse
import numpy as np
import tensorflow as tf
import sys
from time import time
from sklearn.metrics import confusion_matrix

def train_model(context_model_file, data_directory, training_epochs, **args):

    print ('Training for ' + str(training_epochs))

    ## Prepare training data
    img_height = 224
    img_width = 224
    batch_size = 16

    if K.image_data_format() == 'channels_first':
      input_shape = (3, img_width, img_height)
    else:
      input_shape = (img_width, img_height, 3)

    # Directories used
    train_data_dir = data_directory + '\\train'
    validation_data_dir = data_directory + '\\validation'
    test_data_dir = data_directory + '\\test'

    # Prepare data generators
    train_datagen = ImageDataGenerator(rescale = 1./255,
            shear_range = 0.2,
            zoom_range = 0.2,
            horizontal_flip = True)
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True)
    validation_datagen = ImageDataGenerator(rescale = 1./255)
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size)
    classes = len(train_generator.class_indices)

    ## Give same input to both models
    model_input = Input(shape = input_shape)

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

    ## Merge/concatenate outputs of the two sub-models
    context_output = context_model.output
    object_output = object_model.output
    y = concatenate([context_output, object_output])

    ## Add extra layer(s) for classification of activities
    y = Dense(1024, activation = 'relu')(y)
    predictions = Dense(classes, activation = 'softmax')(y)

    ## Set C-CNN to be not trainable
    for layer in context_model.layers:
        layer.trainable = False

    ## Compile model
    model = Model(inputs = model_input, outputs = predictions)
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.summary()

    prefix = 'oc-cnn'
    ## Define callbacks
    tensorboard = TensorBoard(log_dir='logs/' + prefix + "-{}".format(time()))
    es = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3, verbose = 0, mode = 'auto')

    ## Train model
    model.fit_generator(
        train_generator,
        epochs = training_epochs,
        validation_data = validation_generator,
        callbacks = [tensorboard, es])

    # Generate confusion matrix
    print ('Generating confusion matrix..')
    test_datagen = ImageDataGenerator(rescale = 1./255)
    test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size = (img_width, img_height),
        batch_size = 1,
        shuffle = False)
    test_elements = test_generator.__len__()
    probabilities = model.predict_generator(test_generator, test_elements)
    y_pred = np.argmax(probabilities, axis = 1)
    y_true = np.array([], dtype = int)
    y_true = test_generator.classes
    mat = np.matrix(confusion_matrix(y_true, y_pred))
    confusion_file = 'confusion-' + prefix + '-{}'.format(time()) + '.txt';
    with open(confusion_file,'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%d')
    print ('Wrote confusion matrix to ' + confusion_file)

    ## Save model # Save model weights
    model_filename = 'activity-recognition-model.h5'
    model.save(model_filename)
    print ('Saved model to :' + model_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--data_directory',
      help='Local path to training data',
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