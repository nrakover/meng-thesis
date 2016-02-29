import matlab.engine
import shutil
import os

def compileExamples(source_dir, config_file):
	eng = matlab.engine.start_matlab()

	# eng.addpath('/local/nrakover/meng/', nargout=0)

	config = open(config_file, 'r')
	examples = config.readlines()
	config.close()


	for example in examples:
		example_name = example.split('\t')[0]
		if example_name[0:4] == 'MVI_' and example_name[-1].isalpha():
			print('Processing ' + example_name + ':')

			if not os.path.exists(source_dir + example_name):
				os.makedirs(source_dir + example_name)
				print('\t==> created new dir')

			vid_name = example_name[:-1]
			tracks = example.split('\t')[1][:-1].split(',')
			track_paths = [source_dir + vid_name + '/' + t + '.mat' for t in tracks]

			detections_struct_path = source_dir + vid_name + '/detections.mat'
			new_detections_path = source_dir + example_name + '/detections.mat'

			eng.combineTracksIntoDetectionsFile( detections_struct_path, track_paths, new_detections_path, nargout=0)
			print('\t==> consolidated tracks')

			downsampled_vid_path = source_dir + vid_name + '/video.avi'
			new_downsampled_vid_path = source_dir + example_name + '/video.avi'
			assert os.path.exists(downsampled_vid_path)
			shutil.copyfile(downsampled_vid_path, new_downsampled_vid_path)
			print('\t==> copied video')

			opticalflow_path = source_dir + vid_name + '/opticalflow.t7'
			new_opticalflow_path = source_dir + example_name + '/opticalflow.t7'
			assert os.path.exists(opticalflow_path)
			shutil.copyfile(opticalflow_path, new_opticalflow_path)
			print('\t==> copied opticalflow')


	eng.quit()


compileExamples('/local/nrakover/meng/datasets/video-corpus/single_track_videos/', 'pickup_single_track_ADDITIONAL_NEG_config.txt')

