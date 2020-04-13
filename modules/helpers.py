import cv2
import numpy as np

def init_params(stream, config):
	W = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
	H = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
	FPS = stream.get(cv2.CAP_PROP_FPS)

	if W > 1024:
		aspect = 1024 / W
		H = int(H * aspect)
		W = 1024
	
	F = config["focal"]
	K = np.array([[F, 0.0, W // 2],
				  [0.0, F, H // 2],
				  [0.0, 0.0, 1.0]])

	return W, H, FPS, F, K

