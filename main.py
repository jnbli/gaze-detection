# Code adapted from Pysource
# https://pysource.com/2019/01/17/eye-gaze-detection-2-gaze-controlled-keyboard-with-python-and-opencv-p-4/

import cv2
import numpy as np
import dlib
from math import hypot, sqrt

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


def midpoint(p1, p2):
	return int((p1.x+p2.x)/2), int((p1.y+p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks):		
	left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
	right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
	center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
	center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

	# hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
	# ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

	hor_line_length = hypot((left_point[0]-right_point[0]), (left_point[1]-right_point[1]))
	ver_line_length = hypot((center_top[0]-center_bottom[0]), (center_top[1]-center_bottom[1]))

	ratio = hor_line_length/ver_line_length
	return ratio

def get_gaze_ratio(eye_points, facial_landmarks):
	
	left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
						(facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
						(facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
						(facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
						(facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
						(facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
	# cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)


	height, width, _ = frame.shape
	mask = np.zeros((height, width), np.uint8)
	cv2.polylines(mask, [left_eye_region], True, 255, 2)
	cv2.fillPoly(mask, [left_eye_region], 255)
	left_eye = cv2.bitwise_and(gray, gray, mask=mask)		

	min_x = np.min(left_eye_region[:, 0])
	max_x = np.max(left_eye_region[:, 0])
	min_y = np.min(left_eye_region[:, 1])
	max_y = np.max(left_eye_region[:, 1])

	gray_eye = left_eye[min_y: max_y, min_x: max_x]
	_, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)

	height, width = threshold_eye.shape

	left_side_threshold = threshold_eye[0: height, 0: int(width/2)]
	left_side_white = cv2.countNonZero(left_side_threshold)


	right_side_threshold = threshold_eye[0: height, int(width/2): width]
	right_side_white = cv2.countNonZero(right_side_threshold)
	
	gaze_ratio = left_side_white/right_side_white if right_side_white != 0 else 0

	return gaze_ratio

while True:
	_, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = detector(gray)
	for face in faces:
		# x, y = face.left(), face.top()
		# x1, y1 = face.right(), face.bottom()
		# cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)

		landmarks = predictor(gray, face)

		# for landmark_pt in range(0, 68):
		# 	x = landmarks.part(landmark_pt).x
		# 	y = landmarks.part(landmark_pt).y
		# 	cv2.circle(frame, (x, y), 3, (0, 0, 255), 2)

		# Detect Blinking
		left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks)
		right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks)
		blinking_ratio = (left_eye_ratio + right_eye_ratio)/2

		if blinking_ratio > 5.8:
			 cv2.putText(frame, "BLINKING", (50, 150), font, 7, (255, 0, 0))

		
		# Gaze Detection
		gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
		gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)


		gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye) / 2

		
		if gaze_ratio < 0.6:
			cv2.putText(frame, "EYING LEFT", (50, 100), font, 2, (0, 0, 255), 3)
		elif 0.6 <= gaze_ratio and gaze_ratio <= 1.6:
			cv2.putText(frame, "EYING CENTER", (50, 100), font, 2, (0, 0, 255), 3)
		elif gaze_ratio > 1.6:
			cv2.putText(frame, "EYING RIGHT", (50, 100), font, 2, (0, 0, 255), 3)

		cv2.putText(frame, str(gaze_ratio), (50, 200), font, 2, (0, 0, 255), 3)


		nose_tip_mark = landmarks.part(30)
		left_cheek_mark = landmarks.part(2)
		right_cheek_mark = landmarks.part(14)
		left_cheek_dist = sqrt((nose_tip_mark.x - left_cheek_mark.x)**2 + (nose_tip_mark.y - left_cheek_mark.y)**2)
		right_cheek_dist = sqrt((nose_tip_mark.x - right_cheek_mark.x)**2 + (nose_tip_mark.y - right_cheek_mark.y)**2)
		cheek_turn_ratio = left_cheek_dist/right_cheek_dist

		if cheek_turn_ratio < 0.3:
			cv2.putText(frame, "FACING LEFT", (50, 300), font, 2, (0, 0, 255), 3)
		elif 0.3 <= cheek_turn_ratio <= 3.3:
			cv2.putText(frame, "FACING CENTER", (50, 300), font, 2, (0, 0, 255), 3)
		elif cheek_turn_ratio > 3.3:
			cv2.putText(frame, "FACING RIGHT", (50, 300), font, 2, (0, 0, 255), 3)

		cv2.putText(frame, str(cheek_turn_ratio), (50, 400), font, 2, (0, 0, 255), 3)


	cv2.imshow("Frame", frame)


	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()