import cv2
import tqdm
import numpy as np

def final_editing(videos, tracking_results, total_score_list, shot, min_frame, output_file):
	cam = []
	for i in range(len(videos)):
		cam.append(cv2.VideoCapture(videos[i]))

	fps = cam[0].get(cv2.CAP_PROP_FPS)
	width = int(cam[0].get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cam[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
	totalFrames = int(cam[0].get(cv2.CAP_PROP_FRAME_COUNT))

	center_save = []
	center_adjustment_save = []
	for i in range(len(cam)):
		center_adjustment_save.append([])
		center_save.append([])

	prev_shot = None
	prev_cam = -1
	threshold = 0.1
	frame_id = 0
	videoWriter = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))

	count = 0
	mf = min_frame
	for i in tqdm.tqdm(range(int(totalFrames))):
		frame = []
		center_list = []
		for j in range(len(cam)):
			success, f = cam[j].read()
			frame.append(f)

			scale = 1.1
			center = None
			if len(tracking_results[j]) > i:
				line = tracking_results[j][i]
				temp = [int(float(line[3])), int(float(line[2]))]
				center_adjustment = [int(float(line[5])/8), int(float(line[4])/3)]
				center_save[j].append(temp)
				center_adjustment_save[j].append(center_adjustment)
				if len(center_save[j]) > 40 :
					mean = np.sum(center_save[j], axis=0) / len(center_save[j])
					mean_adjust = np.sum(center_adjustment_save[j], axis=0) / len(center_adjustment_save[j])
					local_mean = np.sum(center_save[j][-40:], axis=0) / 40
					local_mean_adjust = np.sum(center_adjustment_save[j][-40:], axis=0) / 40
				else :
					mean = np.sum(center_save[j], axis=0) / len(center_save[j])
					mean_adjust = np.sum(center_adjustment_save[j], axis=0) / len(center_adjustment_save[j])
					local_mean = temp 
					local_mean_adjust = center_adjustment
				center = [int(0.6*local_mean[0]+0.3*mean[0]+0.1*int(float(line[3]))) + int(0.15*int(float(line[5])/8)+0.5*local_mean_adjust[0]+0.35*mean_adjust[0]), int(0.10*mean[1]+0.85*local_mean[1]+0.05*int(float(line[2]))) + int(0.1*int(float(line[4])/3)+0.5*local_mean_adjust[1]+0.4*mean_adjust[1])]    
			
			center_list.append(center)

		curr_score = float('-inf')

		if min_frame == mf:
			for j in range(len(cam)):
				score = sum(total_score_list[j][frame_id:frame_id+mf]) / mf
				if j == prev_cam:
					score += threshold
				if score > curr_score:
					curr_score = score
					curr_cam = j
		elif min_frame <= 0:
			for j in range(len(cam)):
				score = sum(total_score_list[j][frame_id:frame_id+mf]) / mf
				if j == prev_cam:
					score += threshold
				if score > curr_score:
					curr_score = score
					curr_cam = j
			if curr_cam != prev_cam:
				min_frame = mf
		min_frame -= 1
		
		if prev_cam != curr_cam:
			count += 1
		if count <= len(shot):
			read_shot = shot[(count % len(shot)) - 1]
			scale = 1.1
			while 'None' == read_shot or 'Hand' == read_shot:
				count += 1
				read_shot = shot[count]
			if read_shot == 'Close up':
				scale = 5
			elif read_shot == 'Medium Close up':
				scale = 4
			elif read_shot == 'Medium Shot':
				scale = 3
			elif read_shot == 'Medium Long shot':
				scale = 2
				
			if count == 1:
				scale = 1.1
				
			scale_cal = scale * 2
			max_centerX = height-int(height*(1/scale_cal))
			max_centerY = width-int(width*(1/scale_cal))
			min_centerX = int(height*(1/scale_cal))
			min_centerY = int(width*(1/scale_cal))
			if center_list[curr_cam][1] > max_centerY : 
				center_list[curr_cam][1] = max_centerY
			elif center_list[curr_cam][1] < min_centerY :
				center_list[curr_cam][1] = min_centerY

			if center_list[curr_cam][0] > max_centerX :
				center_list[curr_cam][0] = max_centerX
			elif center_list[curr_cam][0] < min_centerX :
				center_list[curr_cam][0] = min_centerX
			
			radiusX,radiusY = int(height/scale_cal),int(width/scale_cal)        
	  
			minX, maxX = max(center_list[curr_cam][0]-radiusX, 0), min(center_list[curr_cam][0]+radiusX, height)
			minY, maxY = max(center_list[curr_cam][1]-radiusY, 0), min(center_list[curr_cam][1]+radiusY, width)
			
			try:
				cropped = frame[curr_cam][minX:maxX, minY:maxY]
			except:
				continue
			
			cropped_resized = cv2.resize(cropped, (width , height))
			videoWriter.write(cropped_resized)
		
		else:
			videoWriter.write(frame[curr_cam])
		
		prev_cam = curr_cam
		frame_id += 1
		
	videoWriter.release()
	print("Editing is done.")