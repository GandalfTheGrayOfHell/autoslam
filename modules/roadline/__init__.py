import cv2
import numpy as np

class RoadLineDetector(object):
	def __init__(self, config, shape):
		self.drawRoadLine = config["drawRoadLine"]
		self.drawRoadRegion = config["drawRoadRegion"]

		self.previous_a_right, self.previous_b_right = 0, 0
		self.previous_a_left, self.previous_b_left = 0, 0

		self.TopLeftX, self.TopLeftY = 0, 0
		self.BottomLeftX, self.BottomRightX = 0, 0
		self.TopRightX, self.TopRightY = 0, 0

		self.W, self.H = shape

		self.bottom_left = [int(self.W * 0.22), int(self.H * 1)]
		self.top_left = [int(self.W * 0.50), int(self.H * 0.65)]
		self.bottom_right = [int(self.W * 1), int(self.H * 1)]
		self.top_right = [int(self.W * 0.70), int(self.H * 0.65)]
	
	def draw(self, image):
		if self.drawRoadLine == 1:
			cv2.line(image, (self.TopLeftX, self.TopLeftY), (self.BottomLeftX, image.shape[0]), (255, 0, 0), 4)
			cv2.line(image, (self.TopRightX, self.TopRightY), (self.BottomRightX, image.shape[0]), (255, 0, 0), 4)

			top = (self.TopLeftX + int((self.TopRightX - self.TopLeftX) / 2), self.TopLeftY)
			bottom = (self.BottomLeftX + int((self.BottomRightX - self.BottomLeftX) / 2), image.shape[0])
			cv2.line(image, top, bottom, (255, 255, 255), 1)

			# verticle line with offset
			# half = (image.shape[1] // 2) + 80
			# cv2.line(image, (half, 0), (half, self.W), (255, 255, 0), 2)

			if self.drawRoadRegion == 1:
				window_img = np.zeros_like(image)
				polyfill = np.array([self.bottom_left, self.bottom_right, self.top_right, self.top_left])
				cv2.fillPoly(window_img, pts=[polyfill], color=(0, 255, 0))
				image = cv2.addWeighted(image, 1, window_img, 0.3, 0)

		return image

	def input(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(gray, 100, 150)
		blur_gray = cv2.GaussianBlur(edges, (7, 7), 0) # useless

		smoothing = 0.95

		vertices = np.array([[self.bottom_left, self.top_left, self.top_right, self.bottom_right]], dtype=np.int32)
		interested = self.region_of_interest(edges, vertices)

		lines = cv2.HoughLinesP(interested, 1, 0.03490658503988659, 10, np.array([]), minLineLength=10, maxLineGap=8) # np.pi / 90
		
		left_sides = { 'x': [], 'y': [] }
		right_sides = { 'x': [], 'y': [] }

		half = (image.shape[1] // 2) + 80
		
		for lane in lines:
			for x1, y1, x2, y2 in lane:
				if x1 < half:
					left_sides['x'].append(x2)
					left_sides['y'].append(y2)
				else:
					right_sides['x'].append(x1)
					right_sides['y'].append(y1)
		
		a_right, b_right = np.polyfit([np.min(right_sides['y']), np.max(right_sides['y'])], [np.min(right_sides['x']), np.max(right_sides['x'])], 1)

		a_left, b_left = np.polyfit([np.max(left_sides['y']), np.min(left_sides['y'])], [np.min(left_sides['x']), np.max(left_sides['x'])], 1)
		
		if self.previous_a_right == 0 and self.previous_b_right == 0:
			self.previous_a_right = a_right
			self.previous_b_right = b_right
		else:
			a_right = self.previous_a_right * smoothing + (1 - smoothing) * a_right
			self.previous_a_right = a_right
			b_right = self.previous_b_right * smoothing + (1 - smoothing) * b_right
			self.previous_b_right = b_right
			
		if self.previous_a_left == 0 and self.previous_b_left == 0:
			self.previous_a_left = a_left
			self.previous_b_left = b_left
		else:
			a_left = self.previous_a_left * smoothing + (1 - smoothing) * a_left
			self.previous_a_left = a_left
			b_left = self.previous_b_left * smoothing + (1 - smoothing) * b_left
			self.previous_b_left = b_left
		
		self.BottomRightX = int(image.shape[0] * a_right + b_right)
		self.BottomLeftX = int(image.shape[0] * a_left + b_left)
		
		self.TopRightX = np.min(right_sides['x'])
		self.TopRightY = np.min(right_sides['y'])
		self.TopLeftX = np.max(left_sides['x'])
		self.TopLeftY = np.min(left_sides['y'])
		
		if self.TopRightY < self.TopLeftY:
			self.TopLeftY = self.TopRightY
			self.TopLeftX = int(self.TopLeftY * a_left + b_left)
		else:
			self.TopRightY = self.TopLeftY
			self.TopRightX = int(self.TopRightY * a_right + b_right)

		
		ratio_road = int((image.shape[1] - (self.BottomRightX - self.BottomLeftX)) / 2)
		steering = (self.BottomLeftX / ratio_road) - 1

		if steering < 0:
			string_steering = 'move to left: %.2fm' % (steering)
		else:
			string_steering = 'move to right: %.2fm' % (steering)

		print(string_steering, flush=False)


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


'''
Want:
[[[ 422 1080]
  [ 960  702]
  [1344  702]
  [1920 1080]]]

'''

# class RoadLineDetector(object):
# 	def __init__(self, config):
# 		self.drawRoadLine = config["drawRoadLine"]
# 		self.previous_a_right, self.previous_b_right = 0, 0
# 		self.previous_a_left, self.previous_b_left = 0, 0

# 	def grayscale(self, img):
# 		return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
# 	def canny(self, img, low_threshold, high_threshold):
# 		return cv2.Canny(img, low_threshold, high_threshold)

# 	def gaussian_blur(self, img, kernel_size):
# 		return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# 	def region_of_interest(self, img, vertices):
# 		mask = np.zeros_like(img)   
# 		if len(img.shape) > 2:
# 			channel_count = img.shape[2]
# 			ignore_mask_color = (255,) * channel_count
# 		else:
# 			ignore_mask_color = 255
			 
# 		cv2.fillPoly(mask, vertices, ignore_mask_color)
# 		masked_image = cv2.bitwise_and(img, mask)
# 		return masked_image

# 	def input(self, image):
# 		gray = self.grayscale(image)
# 		edges = self.canny(gray, 100, 150)
# 		blur_gray = self.gaussian_blur(edges, 7)

# 		rows, cols   = image.shape[:2]
# 		smoothing = 0.95
# 		bottom_left  = [int(cols*0.22), int(rows*1)]
# 		top_left     = [int(cols*0.50), int(rows*0.65)]
# 		bottom_right = [int(cols*1), int(rows*1)]
# 		top_right    = [int(cols*0.70), int(rows*0.65)]
# 		vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
# 		interested = self.region_of_interest(edges, vertices)

# 		self.lines = cv2.HoughLinesP(interested, 1, np.pi/90, 10, np.array([]), minLineLength=10, maxLineGap=8)

# 		right_sides = { 'x': [], 'y': [] }
# 		left_sides = { 'x': [], 'y': [] }
# 		half = (image.shape[1] // 2) + 100
		
# 		for lane in self.lines:
# 			for x1,y1,x2,y2 in lane:
# 				if x1 < half:
# 					left_sides['x'].append(x2)
# 					left_sides['y'].append(y2)
# 				else:
# 					right_sides['x'].append(x1)
# 					right_sides['y'].append(y1)
		
# 		a_right, b_right = np.polyfit([np.min(right_sides['y']), np.max(right_sides['y'])], [np.min(right_sides['x']), np.max(right_sides['x'])], 1)

# 		a_left, b_left = np.polyfit([np.max(left_sides['y']), np.min(left_sides['y'])], [np.min(left_sides['x']) ,np.max(left_sides['x'])], 1)
		
# 		if self.previous_a_right == 0 and self.previous_b_right == 0:
# 			self.previous_a_right = a_right
# 			self.previous_b_right = b_right
# 		else:
# 			a_right = self.previous_a_right * smoothing + (1 - smoothing) * a_right
# 			self.previous_a_right = a_right
# 			b_right = self.previous_b_right * smoothing + (1 - smoothing) * b_right
# 			self.previous_b_right = b_right
			
# 		if self.previous_a_left == 0 and self.previous_b_left == 0:
# 			self.previous_a_left = a_left
# 			self.previous_b_left = b_left
# 		else:
# 			a_left = self.previous_a_left * smoothing + (1 - smoothing) * a_left
# 			self.previous_a_left = a_left
# 			b_left = self.previous_b_left * smoothing + (1 - smoothing) * b_left
# 			self.previous_b_left = b_left
		
# 		self.BottomRightX = int(image.shape[0] * a_right + b_right)
# 		self.BottomLeftX = int(image.shape[0] * a_left + b_left)
		
# 		self.TopRightX = np.min(right_sides['x'])
# 		self.TopRightY = np.min(right_sides['y'])
# 		self.TopLeftX = np.max(left_sides['x'])
# 		self.TopLeftY = np.min(left_sides['y'])
		
# 		if self.TopRightY < self.TopLeftY:
# 			self.TopLeftY = self.TopRightY
# 			self.TopLeftX = int(self.TopLeftY * a_left + b_left)
# 		else:
# 			self.TopRightY = self.TopLeftY
# 			self.TopRightX = int(self.TopRightY * a_right + b_right)
						
# 		ratio_road = int((image.shape[1] - (self.BottomRightX - self.BottomLeftX)) / 2)
# 		steering = (self.BottomLeftX / ratio_road) - 1

# 		if steering < 0:
# 			string_steering = '%.2f'%(steering)
# 		else:
# 			string_steering = '%.2f'%(steering)

# 		print(string_steering)
		
# 		return image

# 	def draw(self, image):
# 		if self.drawRoadLine == 1:
# 			top = (self.TopLeftX + int((self.TopRightX - self.TopLeftX) / 2), self.TopLeftY)
# 			bottom = (self.BottomLeftX + int((self.BottomRightX - self.BottomLeftX)/2), image.shape[0])
# 			cv2.line(image, (self.TopLeftX, self.TopLeftY), (self.BottomLeftX, image.shape[0]), (255, 0, 0), 8)
# 			cv2.line(image, (self.TopRightX, self.TopRightY), (self.BottomRightX, image.shape[0]), (255, 0, 0), 8)
# 			cv2.line(image, top, bottom, (255,255,255), 1)