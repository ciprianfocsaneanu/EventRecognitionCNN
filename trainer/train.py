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

import argparse
import numpy as np
import tensorflow as tf
import sys
from time import time
from sklearn.metrics import confusion_matrix

def train_model(data_directory, job_dir, cnn_model, **args):
    print (data_directory)
    print (job_dir)
    print (cnn_model)

    img_height = 224
    img_width = 224
    batch_size = 25
    first_training_epochs = 1
    second_training_epochs = 1

    # Directories used
    train_data_dir = data_directory + '\\train'
    validation_data_dir = data_directory + '\\validation'
    test_data_dir = data_directory + '\\test'
    print ('Train data dir: ' + train_data_dir)

    if K.image_data_format() == 'channels_first':
      input_shape = (3, img_width, img_height)
    else:
      input_shape = (img_width, img_height, 3)

    print ('After input shape')
    # Create data generators
    train_datagen = ImageDataGenerator(rescale = 1./255,
            # shear_range = 0.2,
            # zoom_range = 0.2,
            # horizontal_flip = True
            )

    print ('After generators0')
    validation_datagen = ImageDataGenerator(rescale = 1./255)

    print ('After generators1')
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        shuffle = True)

    print ('After generators2')
    classes = len(train_generator.class_indices)

    print ('After generators3')
    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size)

    print ('After generators4')

    # Create the base pre-trained model
    if cnn_model in 'inceptionv3':
        base_model = InceptionV3(weights ='imagenet', include_top = False, input_shape = input_shape)
        prefix = 'inceptionv3'
        print ('Base model: InceptionV3')
    elif cnn_model in 'resnet50':
        base_model = ResNet50(weights = 'imagenet', include_top = False, input_shape = input_shape)
        prefix = 'resnet50'
        print ('Base model: ResNet50')
    elif cnn_model in 'densenet121':
        base_model = DenseNet121(weights = 'imagenet', include_top = False, input_shape = input_shape, input_tensor = None)
        prefix = 'densenet121'
        print ('Base model: DenseNet121')
    else:
        print ('Error: ' + model + ' is not a known model')
        sys.exit()

    # Add a global spatial average pooling layer
    x = base_model.output
    if cnn_model in 'resnet50':
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
    # tensorboard = TensorBoard(log_dir='logs/' + prefix + "-{}".format(time()))

    # Train the model on the new data for a few epochs
    model.fit_generator(
        train_generator,
        epochs = first_training_epochs,
        validation_data = validation_generator,
        callbacks = [])

    print ('Finished training only the extra layers')

    # Freeze only subset of base model's layers
    if cnn_model in 'inceptionv3':
        layers_not_trainable = 249
    elif cnn_model in 'resnet50':
        layers_not_trainable = 120
    elif cnn_model in 'densenet121':
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
        epochs = second_training_epochs,
        validation_data = validation_generator,
        callbacks = [checkpoint, es])

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
    for class_index in range(0, classes):
        for index in range(0, int(test_elements/classes)):
            y_true = np.append(y_true, class_index)
    mat = np.matrix(confusion_matrix(y_true, y_pred))
    with open('confusion.txt','wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%d')
    print ('Wrote confusion matrix to: confusion.txt')

    # Save the confusion matrix to the Cloud Storage bucket's jobs directory
    with file_io.FileIO('confusion.txt', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/confusion.txt', mode='w+') as output_f:
            output_f.write(input_f.read())

    # Save the model locally
    model.save('model.h5')

    # Save the model to the Cloud Storage bucket's jobs directory
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())




if __name__ == '__main__':

    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()

    parser.add_argument(
      '--data_directory',
      help='Cloud Storage bucket or local path to training data',
      required=True)

    parser.add_argument(
      '--job-dir',
      help='Cloud Storage bucket to export the model and store temp files',
      required=True)

    parser.add_argument(
      '--cnn_model',
      help='Model to train (InceptionV3/DenseNet121/ResNet50)',
      default='inceptionv3')

    args = parser.parse_args()
    arguments = args.__dict__

    print(arguments)

    train_model(**arguments)