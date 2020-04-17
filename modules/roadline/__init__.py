import cv2
import numpy as np

class RoadLineDetector(object):
	def __init__(self, args):
		self.drawRoadLine = args["drawRoadLine"]

	def input(self, shared_image, shape):
		image = np.frombuffer(shared_image.get_obj(), dtype=np.uint8)
		image.shape = shape
		b_w = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		canny = cv2.Canny(image, 50, 150)
		gaussian_7 = cv2.GaussianBlur(image, (7, 7), 0)
		color_edges = np.dstack((gaussian_7, gaussian_7, gaussian_7))

		rows, cols   = image.shape[:2]
		bottom_left  = [int(cols*0.02), int(rows*1)]
		top_left     = [int(cols*0.35), int(rows*0.65)]
		bottom_right = [int(cols*0.98), int(rows*1)]
		top_right    = [int(cols*0.65), int(rows*0.65)]

		vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
		interested = self.region_of_interest(image, vertices)

		lines = cv2.HoughLinesP(cv2.cvtColor(interested, cv2.COLOR_RGB2GRAY), 1, 0.03490658503988659, 5, np.array([]), minLineLength=10, maxLineGap=8) # np.pi/90

		if self.drawRoadLine == 1:
			for line in lines:
				for x1, y1, x2, y2 in line:
					cv2.line(image, (x1, y1), (x2, y2), [255, 0, 0], 2)

		cv2.imshow("inside", image)
		cv2.waitKey(-1)

		right_sides = { 'x': [], 'y': [] }
		left_sides = { 'x' :[], 'y': [] }

		# TODO: improve this logic later
		half = image.shape[1] // 2

		for lane in lines:
			for x1,y1,x2,y2 in lane:
				if x1 < half:
					left_sides['x'].append(x2)
					left_sides['y'].append(y2)
				else:
					right_sides['x'].append(x1)
					right_sides['y'].append(y1)

		a_right, b_right = np.polyfit([np.min(right_sides['y']), np.max(right_sides['y'])],
									  [np.min(right_sides['x']), np.max(right_sides['x'])], 1)
		a_left, b_left = np.polyfit([np.max(left_sides['y']), np.min(left_sides['y'])],
									[np.min(left_sides['x']), np.max(left_sides['x'])], 1)

		BottomRightX = int(image.shape[0] * a_right + b_right)
		BottomLeftX = int(image.shape[0] * a_left + b_left)

		TopRightX = np.min(right_sides['x'])
		TopRightY = np.min(right_sides['y'])
		TopLeftX = np.max(left_sides['x'])
		TopLeftY = np.min(left_sides['y'])

		if TopRightY < TopLeftY:
			TopLeftY = TopRightY
		else:
			TopRightY = TopLeftY

		top = (TopLeftX + int((TopRightX - TopLeftX) / 2), TopLeftY)
		bottom = (BottomLeftX + int((BottomRightX - BottomLeftX) / 2), image.shape[0])

		ratio_road = int((image.shape[1]-(BottomRightX-BottomLeftX))/2)
		steering = (BottomLeftX / ratio_road) - 1

		if steering < 0:
			string_steering = 'move to left: %.2fm'%(steering)
		else:
			string_steering = 'move to right: %.2fm'%(steering)

		print(string_steering)


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