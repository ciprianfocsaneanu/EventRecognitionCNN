'''
Author: Focsaneanu Andrei-Ciprian
'''
import requests

# Initialize the Keras REST API endpoint URL along with the input image path
KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "../newWIDER/test/Concerts/85.jpg"

# Load the input image and construct the payload for the request
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# Submit the request
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# Ensure the request was sucessful
if r["success"]:
	# loop over the predictions and display them
	for (i, result) in enumerate(r["predictions"]):
		print("{}. {}: {:.4f}".format(i + 1, result["label"], result["probability"]))
else:
	print("Request failed")