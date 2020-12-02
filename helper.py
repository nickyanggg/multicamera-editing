from copy import deepcopy

def tracking_adjust(filename):
	threshold = 20
	result = []
	f = open(filename,'r')
	
	start = True
	line2 = f.readline()
	line2 = line2.split(',')[:6]
	line2.append(1)
	while True:
		line1 = line2
		line2 = f.readline()
		if not line2:
			break
		line2 = line2.split(',')[:6]
		line2.append(1)
		
		if start and int(line1[0]) != 1:
			for i in range(1, int(line1[0])):
				tmp = deepcopy(line1)
				tmp[0] = str(i)
				if int(line1[0]) - i <= threshold:
					tmp[-1] = 0.8
					tmp[-1] = 1
				else:
					tmp[-1] = 0
				result.append(tmp)
			result.append(line1)
			start = False

		gap = int(line2[0]) - int(line1[0])
		for i in range(1, gap):
			if i < gap/2:
				tmp = deepcopy(line1)
			else:
				tmp = deepcopy(line2)
			tmp[0] = str(int(line1[0]) + i)
			if i <= threshold or gap - i <= threshold:
				tmp[-1] = 0.8
				tmp[-1] = 1
			else:
				tmp[-1] = 0
			result.append(tmp)
		
		result.append(line2)
	f.close()
	return result

def get_smooth_frame(tempo):
	min_frame = 30
	max_frame = 80
	# 60 -> 80, 110 -> 30
	smooth_frame = (-1) * tempo + 140
	smooth_frame = max(min_frame, smooth_frame)
	smooth_frame = min(max_frame, smooth_frame)
	return int(smooth_frame)

def get_shot_only(txtfilename) :
	all_shot = [str(line.strip()) for line in open(txtfilename)]
	if "smooth" in txtfilename :
		smooth_TXT_name = txtfilename.split('_smooth')[0].split('/')[-1] + '.txt'
	else :
		smooth_TXT_name = txtfilename.split('/')[-1]
	folder = txtfilename.replace(str(txtfilename.split('/')[-1]), '')
	smooth_TXT_name = folder + smooth_TXT_name

	shot = []
	shot.append(all_shot[0])
	for i in range(1, len(all_shot)):
		if all_shot[i] == all_shot[i-1]:
			continue
		shot.append(all_shot[i])
	return shot



