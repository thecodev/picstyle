import cv2
import os

def resize(image, width=None, height=None):
	if (width is None) & (height is None):
		raise Exception("Height and Width npth are None")
	elif (width is not None) & (height is not None):
		raise Exception("You haved passed npth Height and Width both value")
	elif (width is not None) & (height is None):
		h, w, c = image.shape
		height = int((h / w) * width)
		return cv2.resize(image, (width, height))
	elif (width is None) & (height is not None):
		h, w, c = image.shape
		width = int((w / h) * height)
		return cv2.resize(image, (width, height))


def loadModel(name):
	net = cv2.dnn.readNetFromTorch(os.path.join("models", name))
	return net 


def preprocess(image):
	image = resize(image, width=600)
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)

	return blob

def predict(net, blob):
	net.setInput(blob)
	output = net.forward()
	output = output.reshape((3, output.shape[2], output.shape[3]))
	output[0] += 103.939
	output[1] += 116.779
	output[2] += 123.680
	output /= 255.0
	output = output.transpose(1, 2, 0)

	return output

if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	net = loadModel("starry_night.t7")
	while True:
		ret, frame = cap.read()
		if frame is None:
			break

		blob = preprocess(frame)

		output = predict(net, blob)

		cv2.imshow("frame", output)

		if cv2.waitKey(1) & 0xff==27:
			break

	cap.release()
	cv2.destroyAllWindows()



