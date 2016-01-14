import matlab.engine
import os

SOURCE_DIR = '/local/nrakover/meng/datasets/video-corpus/20121214-new3-corpus/'


def doAllMATLABPreprocessing(source_dir, destination_dir, frame_downsample_rate, size_downsample_rate, num_proposals):
	'''
	IMPORTANT: this MUST be run from /local/nrakover/meng/objectness/object-proposals/
	'''
	eng = matlab.engine.start_matlab()

	eng.initialize(nargout=0)
	eng.addpath('/local/nrakover/meng/', nargout=0)

	source_dir_contents = os.listdir(source_dir)
	for filename in source_dir_contents:
		if filename.endswith('.mov'):
			vid_name = filename[:-4]
			print('Processing ' + vid_name + ':')

			vid_path = source_dir + filename
			downsampled_vid_path = destination_dir + vid_name + '/video.avi'
			eng.downsampleVideo(vid_path, downsampled_vid_path, frame_downsample_rate, size_downsample_rate, nargout=0)
			print('\t==> downsampling done')

			detections_path = destination_dir + vid_name + '/detections.mat'
			eng.videoToDetections(downsampled_vid_path, detections_path, num_proposals, nargout=0)
			print('\t==> proposals generation done')

	eng.quit()


doAllMATLABPreprocessing(SOURCE_DIR, '/local/nrakover/meng/datasets/video-corpus/videos_272x192_all-frames_75-proposals/', 1, 0.4, 75)

