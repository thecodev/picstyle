from flask import Flask
from PIl import Image


def convertToBinaryData(filename):
	#Convert digital data to binary format
	with open(filename, 'rb') as file:
		blobData = file.read()
	return blobData


# pil_image = Image.open(image)