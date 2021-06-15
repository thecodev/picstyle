from PIL import Image
from flask import Flask, render_template, Response, request, jsonify, make_response
import cv2
from neuralStyle import loadModel, preprocess, predict
import numpy as np 
import base64, io, os
app = Flask(__name__)

images = os.listdir("static/images")
models = os.listdir("models")

displayModelImages = []
for model in models:
	for img in images:
		if model.split(".")[0] in img:
			displayModelImages.append(img)
print(displayModelImages)

def convertToBinaryData(filename):
	#Convert digital data to binary format
	with open(filename, 'rb') as file:
		blobData = file.read()
	return blobData

@app.route('/', methods=['POST', "GET"])
def index():
	return render_template('index.html', modelImages=displayModelImages)



@app.route('/capture', methods=['POST', 'GET'])
def capture():
	print("ok")
	model = request.values.get("style").split(".")[0] + ".t7"
	image = request.files["image"]
	try:
		image = Image.open(image)

		image = np.array(image)
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		blob = preprocess(image)

		net = loadModel(model)

		image = predict(net, blob)
		image = cv2.normalize(image,   np.zeros((image.shape[1], image.shape[0])), 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(image)
		data = io.BytesIO()
		image.save(data, "JPEG")
		image = base64.b64encode(data.getvalue())
		print("done")
		return jsonify({'status': True, 'image': image.decode('utf-8')})
	except Exception as e:
		print(e)
		return jsonify({'status': False,})

	



if __name__ == "__main__":
	app.run(debug=False)