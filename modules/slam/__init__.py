import cv2
import numpy as np

from . import frame as fm
from .slam_map import SlamMap, SlamPoint
from .display import Display
from . import helpers as helper


class Slam(object):
	def __init__(self, K, width, height):
		self.K = K
		self.width = width
		self.height = height
		self.slam_map = SlamMap()

	def matchFrame(self, frame1, frame2):
		matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
		matches = matcher.knnMatch(frame1.descriptors, frame2.descriptors, k=2)
		ratio = 0.65
		match_idx1, match_idx2 = [], []
		for m1, m2 in matches:
			if (m1.distance < ratio * m2.distance) and (m1.distance < 32):
				if m1.queryIdx not in match_idx1 and m1.trainIdx not in match_idx2:
					match_idx1.append(m1.queryIdx)
					match_idx2.append(m1.trainIdx)

		# We need at least 8 matches for calculating Fundamental matrix
		assert(len(match_idx1) >= 8)
		assert(len(match_idx2) >= 8)

		assert(len(set(match_idx1)) == len(match_idx1))
		assert(len(set(match_idx2)) == len(match_idx2))

		match_idx1, match_idx2 = np.asarray(match_idx1), np.asarray(match_idx2)

		frame1_pts_un = frame1.keypoints_un[match_idx1, :]
		frame2_pts_un = frame2.keypoints_un[match_idx2, :]

		F, inlier_mask = cv2.findFundamentalMat(frame1_pts_un, frame2_pts_un, cv2.RANSAC, 0.1, 0.99)
		E = self.K.T @ F @ self.K
		inlier_mask = inlier_mask.astype(bool).squeeze()

		_, R, t, _ = cv2.recoverPose(E, frame1_pts_un, frame2_pts_un, self.K)
		pose = np.eye(4)
		pose[0:3, 0:3] = R
		pose[0:3, 3] = t.squeeze()
		return pose, match_idx1[inlier_mask], match_idx2[inlier_mask]

	def input(self, image):
		frame = fm.Frame(image, self.K)
		self.slam_map.addFrame(frame)

		if frame.id == 0:
			return

		curr_frame = self.slam_map.frames[-1]
		prev_frame = self.slam_map.frames[-2]

		relative_pose, match_idx1, match_idx2 = self.matchFrame(prev_frame, curr_frame)

		# Compose the pose of the frame (this forms our pose estimate for optimize)
		curr_frame.pose = relative_pose @ prev_frame.pose
		
		P1 = prev_frame.pose[0:3, :]
		P2 = curr_frame.pose[0:3, :]

		# If a point has been observed in previous frame, add a corresponding observation even in the current frame
		for i, idx in enumerate(match_idx1):
			if prev_frame.map_points[idx] is not None and curr_frame.map_points[match_idx2[i]] is None:
				prev_frame.map_points[idx].addObservation(curr_frame, match_idx2[i])

		# If the map contains points
		sbp_count = 0
		if len(self.slam_map.points) > 0:
			# Optimize the pose only by keeping the map_points fixed
			reproj_error = self.slam_map.optimize(local_window=2, fix_points=True)
			
		# Triangulate only those points that were not found by searchByProjection (innovative points)
		points3d, _ = helper.triangulate(P1, prev_frame.keypoints[match_idx1], P2, curr_frame.keypoints[match_idx2])
		remaining_point_mask = np.array([curr_frame.map_points[i] is None for i in match_idx2])

		add_count = 0
		for i, point3d in enumerate(points3d):
			if not remaining_point_mask[i]:
				continue

			# Check if 3D point is in front of both frames
			point4d_homo = np.hstack((point3d, 1))
			point1 = prev_frame.pose @ point4d_homo
			point2 = curr_frame.pose @ point4d_homo
			if point1[2] < 0 and point2[2] < 0:
				continue

			proj_point1, _ = helper.projectPoints(point3d, prev_frame.pose, self.width, self.height)
			proj_point2, _ = helper.projectPoints(point3d, curr_frame.pose, self.width, self.height)
			
			err1 = np.sum(np.square(proj_point1-prev_frame.keypoints[match_idx1[i]]))
			err2 = np.sum(np.square(proj_point2-curr_frame.keypoints[match_idx2[i]]))

			if err1 > 1 or err2 > 1:
				continue

			color = image[int(np.round(curr_frame.keypoints[match_idx2[i]][1])), int(
				np.round(curr_frame.keypoints[match_idx2[i]][0]))]
			point = SlamPoint(points3d[i], color)
			point.addObservation(prev_frame, match_idx1[i])
			point.addObservation(curr_frame, match_idx2[i])
			self.slam_map.addPoint(point)
			add_count += 1

		if frame.id > 4 and frame.id % 4 == 0:
			self.slam_map.optimize()

		return frame.drawFrame(prev_frame.keypoints_un[match_idx1], curr_frame.keypoints_un[match_idx2])