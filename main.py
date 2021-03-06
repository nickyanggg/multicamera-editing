import os
import librosa
import argparse
from score import *
from helper import *
from editing import final_editing

def main():
	parser = argparse.ArgumentParser(description='''multicamera and style editing''')
	parser.add_argument('--video_dir', type=str, required=True, default='', help='path to video file dir')
	parser.add_argument('--main_cam', type=str, required=False, default='', help='file name of video source from main camera')
	parser.add_argument('--tracking_dir', type=str, required=True, default='', help='path to tracking file dir')
	parser.add_argument('--min_frame', type=int, required=False, default=80, help='minimum duration of a segment')
	parser.add_argument('--auto_frame', action='store_true', help='calculate minimum duration of a segment from bpm if true')
	parser.add_argument('--shot_file', type=str, required=True, default='', help='shot type sequence generated by style editing model')
	parser.add_argument('--output_file', type=str, required=True, default='', help='result of multicamera and style editing')
	args = parser.parse_args()

	video_folder = args.video_dir
	tracking_folder = args.tracking_dir
	main_cam = args.main_cam
	auto = args.auto_frame
	smooth_frame = args.min_frame
	shot_file = args.shot_file
	output_file = args.output_file

	videos = os.listdir(video_folder)
	results = []
	tracking_results = []
	user_preference = []
	for i in range(len(videos)):
		if videos[i] == main_cam:
			user_preference.append(1)
		else:
			user_preference.append(0)
		tr = os.path.splitext(videos[i])[0]
		results.append(os.path.join(tracking_folder, tr + ".txt"))
		videos[i] = os.path.join(video_folder, videos[i])
		tracking_results.append(tracking_adjust(results[-1]))

	total_score_list = calculate_score(videos, tracking_results, user_preference)

	if auto:
		y, sr = librosa.load(videos[1], duration=10)
		tempo, beats = librosa.beat.beat_track(y=y, sr=sr, start_bpm=120)
		smooth_frame = get_smooth_frame(tempo)
	shot = get_shot_only(shot_file)
	final_editing(videos, tracking_results, total_score_list, shot, smooth_frame, output_file)

if __name__ == '__main__':
	main()


