import cv2
import numpy as np

class Darknet(object):
	def __init__(self, args, dim):
		self.labels = open(args["labelsPath"]).read().strip().split("\n")
		self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")
		self.net = cv2.dnn.readNetFromDarknet(args["configPath"], args["weightsPath"])
		self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
		self.ln = self.net.getLayerNames()
		self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
		self.drawDarknet, self.confidence, self.threshold = args["drawDarknet"], args["confidence"], args["threshold"]
		self.W, self.H = dim[0], dim[1]

	def input(self, image):
		self.net.setInput(cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False))
		self.layersOutput = self.net.forward(self.ln)

	def draw(self, image):
		pass