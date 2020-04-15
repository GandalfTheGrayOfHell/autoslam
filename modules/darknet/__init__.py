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
		if self.drawDarknet == 1:
			boxes, confidences, classIDs = [], [], []

			for output in self.layersOutput:
				for detection in output:
					scores = detection[5:]
					classID = np.argmax(scores)
					confidence = scores[classID]


					if confidence > self.confidence:
						box = detection[0:4] * np.array([self.W, self.H, self.W, self.H])
						(cX, cY, w, h) = box.astype("int")

						x = int(cX - (w / 2))
						y = int(cY - (h / 2))

						boxes.append([x, y, int(w), int(h)])
						confidences.append(float(confidence))
						classIDs.append(classID)

			# MAYBE: try replacing for another filter since it slows down.
			# ref: https://medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359

			# TODO: Investigate NMS boxes.
			# Apparantely Yolo already does this
			# ref: https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088
			idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

			if len(idxs) > 0:
				for i in idxs.flatten():
					(x, y) = (boxes[i][0], boxes[i][1])
					(w, h) = (boxes[i][2], boxes[i][3])

					color = [int(c) for c in self.colors[classIDs[i]]]
					cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
					text = "{}: {:.2f}".format(self.labels[classIDs[i]], confidences[i])
					# cv.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
					cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

		return image