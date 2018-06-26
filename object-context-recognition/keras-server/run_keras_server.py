'''
Author: Focsaneanu Andrei-Ciprian
'''
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import keras.backend as K
from PIL import Image
import numpy as np
import flask
from flask_cors import CORS, cross_origin
import io
import argparse

# Initialize our Flask application and the Keras model
app = flask.Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
model = None

# Hardcode WIDER 19 classes
classes = ['Baseball', 'Car Accident', 'Ceremony', 'Concerts', 'Football', 'Gymnastics', 'Hockey', 'Ice Skating', 'Jockey',
'Matador Bullfighter', 'Meeting', 'People marching', 'Picnic', 'Row Boat', 'Soccer', 'Soldier Patrol', 'Spa', 'Surgeons', 'Swimming']

def load_pretrained_model(pretrained_model, **args):
	K.clear_session()
	# Load the pre-trained Keras model
	global model
	print ('Loading model: ' + pretrained_model)
	model = load_model(pretrained_model)
	model._make_predict_function()

def prepare_image(image, target):
	# If the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# Resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = np.divide(image, 255.)

	return image

# Implement POST method
@app.route("/predict", methods=["POST"])
@cross_origin()
def predict():
	# Initialize the data dictionary that will be returned from the
	data = {"success": False}

	# Ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# Read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# Preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224, 224))

			# Classify the input image and then initialize the list
			# of predictions to return to the client
			preds = model.predict(image)
			data["predictions"] = []

			for i in range(0, len(preds[0])):
				r = {"label": classes[i], "probability": float(preds[0][i])}
				data["predictions"].append(r)

			# Take the top-5 predictions
			data["predictions"] = list(reversed(sorted(data["predictions"], key=lambda k: k['probability'])))[:5]

			# Indicate that the request was a success
			data["success"] = True

	# Return the data dictionary as a JSON response
	return flask.jsonify(data)

# If this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
	  '--pretrained_model',
	  help='Local path to trained model',
	  required=True)

	args = parser.parse_args()
	arguments = args.__dict__

	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))

	load_pretrained_model(**arguments)
	app.run()