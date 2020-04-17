import time
import math
import numpy as np
import itertools
import cv2
from . import helpers as helper



class Frame(object):
	newid = itertools.count().__next__
	def __init__(self, image, K):
		self.K = K
		self.K_inv = np.linalg.inv(K)
		self.pose = np.eye(4)
		self.orb = cv2.ORB_create()

		if image is not None:
			self.height, self.width = image.shape[0:2]
			self.id = Frame.newid()
			self.image = image
			self.extractFeatures(image)
			self.map_points = [None]*len(self.keypoints_un)
			
	def extractFeatures(self, image):
		corners = cv2.goodFeaturesToTrack(np.mean(image, axis=2).astype(np.uint8), 2000, qualityLevel=0.02, minDistance=7)
		keypoints = [cv2.KeyPoint(x=c[0][0], y=c[0][1], _size=20) for c in corners]
		keypoints, self.descriptors = self.orb.compute(image, keypoints)
		keypoints =  [(kp.pt[0], kp.pt[1]) for kp in keypoints]
		self.keypoints_un = np.asarray(keypoints)

	def drawFrame(self, points1, points2):
		image = np.copy(self.image)
		if (points1.shape[0] > 0) and (points2.shape[0] > 0):
			for i in range(points1.shape[0]):
				point1 = points1[i, :]
				point2 = points2[i, :]
				cv2.circle(image, (int(point1[0]), int(point1[1])), color=(0, 255, 0), radius=2)
				cv2.arrowedLine(image,(int(point1[0]), int(point1[1])),(int(point2[0]), int(point2[1])), (0, 0, 255), 1, 8, 0, 0.2)
		return image

	@property
	def keypoints(self):
		if not hasattr(self, '_kps'):
			self._kps = helper.normalizePoints(self.keypoints_un, self.K_inv)
		return self._kps