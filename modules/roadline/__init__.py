import cv2
import numpy as np

class RoadLineDetector(object):
	def __init__(self, args):
		pass

	def input(self, image):
		pass

	def draw(self, img):
		pass

	def canny(self, img, low_threshold, high_threshold):
		return cv2.Canny(img, low_threshold, high_threshold)

	def gaussian_blur(self, img, kernel_size):
		return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

	def region_of_interest(self, img, vertices):
		mask = np.zeros_like(img)   
		if len(img.shape) > 2:
			channel_count = img.shape[2]
			ignore_mask_color = (255,) * channel_count
		else:
			ignore_mask_color = 255   
		cv2.fillPoly(mask, vertices, ignore_mask_color)
		masked_image = cv2.bitwise_and(img, mask)
		return masked_image












# import cv2
# import numpy as np

# class RoadLineDetector(object):
# 	def __init__(self, args):
# 		self.drawRoadLine = args["drawRoadLine"]
# 		self.lines = None

# 	def input(self, frame):
# 		canny_image = self.canny_edge_detector(frame)
# 		cv2.imwrite("canny.jpg", canny_image)
# 		cropped_image = self.region_of_interest(canny_image)
# 		cv2.imwrite("cropped.jpg", cropped_image)
# 		self.lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 150, np.array([]), minLineLength=30, maxLineGap=0)
# 		cv2.imwrite("hough.jpg", self.lines)
# 		self.lines = self.average_slope_intercept(frame)
# 		cv2.imwrite("intercept.jpg", self.lines)
			
# 	def canny_edge_detector(self, frame):  
# 		blur = cv2.GaussianBlur(frame, (5, 5), 0)  
# 		canny = cv2.Canny(blur, 50, 150) 
# 		return canny 


# 	def region_of_interest(self, image): 
# 		height = image.shape[0] 
# 		polygons = np.array([[(200, height), (1100, height), (550, 250)]])
# 		mask = np.zeros_like(image)
# 		cv2.fillPoly(mask, polygons, 255)
# 		return cv2.cuda.bitwise_and(image, mask) 

# 	def create_coordinates(self, image, line_parameters):
# 		slope, intercept = line_parameters
		
# 		y1 = image.shape[0] 
# 		y2 = int(y1 * (3 / 5)) 
# 		x1 = int((y1 - intercept) / slope) 
# 		x2 = int((y2 - intercept) / slope) 
# 		return np.array([x1, y1, x2, y2])


# 	def average_slope_intercept(self, image): 
# 		left_fit, right_fit = [], []

# 		for line in self.lines: 
# 			x1, y1, x2, y2 = line.reshape(4) 
# 			parameters = np.polyfit((x1, x2), (y1, y2), 1)  
# 			slope = parameters[0] 
# 			intercept = parameters[1] 
# 			if slope < 0: 
# 				left_fit.append((slope, intercept)) 
# 			else: 
# 				right_fit.append((slope, intercept)) 
				
# 				left_fit_average = np.average(left_fit, axis=0) 
# 				right_fit_average = np.average(right_fit, axis=0) 
# 				left_line = self.create_coordinates(image, left_fit_average) 
# 				right_line = self.create_coordinates(image, right_fit_average) 
# 				return np.array([left_line, right_line])

# 	def draw(self, image):
# 		if self.drawRoadLine == 1:
# 			line_image = np.zeros_like(image) 
# 			if self.lines is not None: 
# 				for x1, y1, x2, y2 in self.lines:
# 					try: 
# 						cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
# 					except OverflowError:
# 						print("[INFO] (road/display_lines) Integer out of bounds")
# 						return cv2.cuda.addWeighted(image, 0.8, line_image, 1, 1)
# 					else:
# 						return image
# 		else:
# 			return image