import cv2
import math
import tqdm
import numpy as np
from copy import deepcopy
from pykalman import KalmanFilter
from mtcnn.mtcnn import MTCNN
from multiprocessing import Pool

def kalman2D(observations):
	if len(observations) < 2:
		return observations
	kf = KalmanFilter(
			initial_state_mean = observations[0],
			transition_matrices = [[1.2, 1],[0, 1]],
			n_dim_obs = 2
		)
	pred_state, state_cov = kf.filter(observations)
	return pred_state.tolist()

def face_orientation(frame, landmarks):
	size = frame.shape
	chin_x = (landmarks['keypoints']['mouth_left'][0] + landmarks['keypoints']['mouth_right'][0] + landmarks['keypoints']['nose'][0])/3
	mouth_y = (landmarks['keypoints']['mouth_left'][1] + landmarks['keypoints']['mouth_right'][1]) / 2
	nose_y = landmarks['keypoints']['nose'][1]
	chin_y = mouth_y * 2 - nose_y

	image_points = np.array([
							(landmarks['keypoints']['nose'][0], landmarks['keypoints']['nose'][1]),     # Nose tip
							(chin_x, chin_y),   # Chin
							(landmarks['keypoints']['left_eye'][0], landmarks['keypoints']['left_eye'][1]),     # Left eye left corner
							(landmarks['keypoints']['right_eye'][0], landmarks['keypoints']['right_eye'][1]),     # Right eye right corne
							(landmarks['keypoints']['mouth_left'][0], landmarks['keypoints']['mouth_left'][1]),     # Left Mouth corner
							(landmarks['keypoints']['mouth_right'][0], landmarks['keypoints']['mouth_right'][1])      # Right mouth corner
						], dtype="double")

	model_points = np.array([
							(0.0, 0.0, 0.0),             # Nose tip
							(0.0, -330.0, -65.0),        # Chin
							(-225.0, 170.0, -135.0),     # Left eye left corner
							(225.0, 170.0, -135.0),      # Right eye right corne
							(-150.0, -150.0, -125.0),    # Left Mouth corner
							(150.0, -150.0, -125.0)      # Right mouth corner
						])

	center = (size[1]/2, size[0]/2)
	focal_length = center[0] / np.tan(60/2 * np.pi / 180)
	camera_matrix = np.array(
						 [[focal_length, 0, center[0]],
						 [0, focal_length, center[1]],
						 [0, 0, 1]], dtype = "double"
						 )

	dist_coeffs = np.zeros((4,1))
	(success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

	axis = np.float32([[500,0,0], 
					   [0,500,0], 
					   [0,0,500]])
						  
	imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
	modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
	rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

	proj_matrix = np.hstack((rvec_matrix, translation_vector))
	eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

	pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

	pitch = math.degrees(math.asin(math.sin(pitch)))
	roll = -math.degrees(math.asin(math.sin(roll)))
	yaw = math.degrees(math.asin(math.sin(yaw)))

	return (str(int(roll)), str(int(pitch)), str(int(yaw)))

def generate_total_score(user_score, center_score, yaw_score):
	return 0.01 * user_score + 0.2 * center_score + yaw_score

def generate_center_score(frame_info, center):
	pos_x = float(frame_info[2]) + float(frame_info[4]) / 2
	dist = abs(pos_x - center)
	score = frame_info[-1] * (center * center - dist * dist)/(center * center)
	return score

def generate_yaw_score(i, videos, tracking_results):
	video = videos[i]
	ret = []
	detector_M = MTCNN()
	cam = cv2.VideoCapture(video)
	fps = cam.get(cv2.CAP_PROP_FPS)
	width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
	totalFrames = cam.get(cv2.CAP_PROP_FRAME_COUNT)

	start_frame_id = 1
	end_frame_id = int(totalFrames)
	obs = []
	prev = None
	for num in tqdm.tqdm(range(start_frame_id, end_frame_id + 1)):
		if num > len(tracking_results[i]):
			ret.append(0)
			continue
		line = tracking_results[i][num - 1]
		minX = int(float(line[2]))
		maxX = minX + int(float(line[4]))
		center = (minX + maxX) / 2 - minX
		
		minY = int(float(line[3]))
		maxY = minY + int(float(line[5]))
		
		success, frame = cam.read()
		if not success:
			ret.append(0)
			prev = None
			continue
		cropped = frame[minY-10:maxY+10, minX-10:maxX+10]
		
		faces = detector_M.detect_faces(cropped[...,::-1])
		dist = float("inf")
		face = None
		for result in faces:
			nose = result['keypoints']['nose']
			if abs(nose[0] - center) < dist:
				face = result
				dist = abs(nose[0] - center)
		
		tmp = deepcopy(cropped)
		if face:
			x, y, w, h = face['box']
			x1, y1 = x + w, y + h
			rotate_degree = face_orientation(cropped, face)
			yaw = float(rotate_degree[-1])
			if prev != None:
				kalman_yaw = kalman2D([prev, [yaw, yaw - prev[0]]])[-1][0]
				if abs(kalman_yaw - yaw) > 20:
					yaw = kalman_yaw
				prev = [yaw, yaw - prev[0]]
			else:
				prev = [yaw, 0]
			ret.append(math.cos(math.radians(yaw)) * tracking_results[i][num-1][-1])
			
		else:
			ret.append(0)
			prev = None
	return ret

def score_interpol(score_list, threshold=60):
	for i in range(len(score_list)):
		left_idx = -1
		count = 0
		for j in range(len(score_list[i])):
			if score_list[i][j] != 0:
				if left_idx > -1 and count <= threshold:
					left_score = score_list[i][left_idx]
					right_score = score_list[i][j]
					gap = float(right_score - left_score) / (j - left_idx)
					for k in range(left_idx + 1, j):
						score_list[i][k] = score_list[i][k-1] + gap   
				count = 0
				left_idx = j
			else:
				count += 1
	return score_list

def calculate_score(videos, tracking_results, user_preference):
	params = []
	for i in range(len(videos)):
		params.append((i, videos, tracking_results))
	p = Pool(processes=len(videos))
	score_list = p.starmap(generate_yaw_score, params)
	p.close
	yaw_score_list = score_interpol(score_list)

	total_score_list = []
	user_score_list = []
	center_score_list = []
	yaw_score_list = [i for sublist in yaw_score_list for i in sublist]

	for i in range(len(tracking_results)):
		cam = cv2.VideoCapture(videos[i])
		width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
		totalFrames = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
		total_score_list.append([])
		for j in range(totalFrames):
			user_score = user_preference[i]
			center_score = 0
			if j < len(tracking_results[i]):
				f = tracking_results[i][j]
				center_score = generate_center_score(f, width / 2)
			yaw_score = yaw_score_list[i*totalFrames + j]
			total_score = generate_total_score(user_score, center_score, yaw_score)
			total_score_list[i].append(total_score)
	print("Scoring is done.")
	return total_score_list

