from PIL import Image
from flask import Flask, render_template, Response, request
import cv2
from neuralStyle import loadModel, preprocess, predict
import numpy as np 
import base64, io

app = Flask(__name__)


def convertToBinaryData(filename):
	#Convert digital data to binary format
	with open(filename, 'rb') as file:
		blobData = file.read()
	return blobData

@app.route('/')
def index():
	return render_template('index.html', predicted=False)

@app.route('/show', methods=['POST', "GET"])
def show():
	image = request.files['img']
	image = Image.open(image)
	image = np.array(image)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	blob = preprocess(image)

	net = loadModel("starry_night.t7")

	image = predict(net, blob)
	image = cv2.normalize(image,   np.zeros((image.shape[1], image.shape[0])), 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = Image.fromarray(image)
	data = io.BytesIO()
	image.save(data, "JPEG")
	image = base64.b64encode(data.getvalue())


	return render_template('index.html', predicted=True, img=image.decode('utf-8'))



# pil_image = Image.open(image)


if __name__ == "__main__":
	app.run(debug=True)