import keras.backend as K
import tensorflow as tf
import numpy as np
from keras.models import load_model, Sequential, model_from_config, Model

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def

# reset session
K.clear_session()
sess = tf.Session()
K.set_session(sess)

# disable loading of learning nodes
K.set_learning_phase(0)

# load model
model = load_model('local-activity-recognition-model.h5')
config = model.get_config()
weights = model.get_weights()
new_Model = Model.from_config(config)
new_Model.set_weights(weights)

# export saved model
export_path = '.' + '/export'
builder = saved_model_builder.SavedModelBuilder(export_path)

signature = predict_signature_def(inputs={'export_input': new_Model.input},
                                  outputs={'export_output': new_Model.output})

with K.get_session() as sess:
    builder.add_meta_graph_and_variables(sess=sess,
                                         tags=[tag_constants.SERVING],
                                         signature_def_map={
                                             signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})
    builder.save()