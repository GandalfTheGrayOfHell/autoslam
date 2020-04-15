import sys
import json
import cv2
import time
import numpy as np


from multiprocessing import Process

from modules.slam import Slam, Display
from modules import helpers as hm
from modules.darknet import Darknet
from modules.roadline import RoadLineDetector

np.set_printoptions(suppress=True)
np.random.seed(101)
cv2.cuda.setDevice(0)

if __name__ == '__main__':
	# config file
	config = json.loads(open("./config.json").read())
	print("[COMPLETE] Config and Setup vars")

	# get video stream
	capture = cv2.VideoCapture(config["path"]) 
	cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

	if capture.isOpened():
		print("[COMPLETE] Video stream")
		W, H, FPS, F, K = hm.init_params(capture, config)
		print("[COMPLETE] Video parameters")
	else:
		print("[FAILED] Video stream")

	slam = Slam(K, W, H)
	disp = Display()
	darknet = Darknet(config, (W, H))
	roadline_detector = RoadLineDetector(config)
	start_time = time.time()

	i = 0
	while capture.isOpened():
		ret, frame = capture.read()

		if ret == True:
			# TODO: Right now GPU resize is slower than CPU resize. Maybe once the workload increase do a performance check
			# UPDATE: the upload and download to gpu is very expensive. not gonna work most likely
			# frame = cv2.cuda.resize(cv2.cuda_GpuMat(frame), (W, H)).download()
			frame = cv2.resize(frame, (W, H))
			
			darknet.input(frame)
			#lines = roadline_detector.input(frame)
			frame = slam.input(frame)
			frame = darknet.draw(frame)
			#roadline_detector.draw(frame)

			if i > 0:
				cv2.imshow('frame', frame)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
			disp.update(slam.slam_map)
		else:
			break
		i += 1

	print("--- %s seconds ---" % (time.time() - start_time))
	disp.finish()
	capture.release()
	cv2.destroyAllWindows()
	sys.exit(0)