import os


SOURCE_DIR = '20121214-new3-corpus/'
# SOURCE_DIR = 'new-corpus/'
# SOURCE_DIR = 'new-corpus3/'


def generateDirectoryForEachVideo(source, destination):
	source_dir_contents = os.listdir(source)

	for filename in source_dir_contents:
		if filename.endswith('.mp4') or filename.endswith('.avi') or filename.endswith('.mov'):
			vid_name = filename[:-4]

			new_dir_path = destination + vid_name
			if not os.path.exists(new_dir_path):
				os.makedirs(new_dir_path)

generateDirectoryForEachVideo(SOURCE_DIR, 'single_track_videos/')

import shutil
def cleanUp1(source, current_dest, new_dest):
	source_dir_contents = os.listdir(source)

	for filename in source_dir_contents:
		if filename.endswith('.mov'):
			vid_name = filename[:-4]

			old_detections = current_dest + vid_name + '/detections.mat'
			new_detections = new_dest + vid_name + '/detections.mat'

			shutil.copyfile(old_detections, new_detections)

			old_features = current_dest + vid_name + '/features.t7'
			new_features = new_dest + vid_name + '/features.t7'

			if os.path.exists(old_features):
				shutil.copyfile(old_features, new_features)


def cleanUp2(source):
	source_dir_contents = os.listdir(source)

	for vid_dir in source_dir_contents:
		vid_dir_contents = os.listdir(source + vid_dir)
		for filename in vid_dir_contents:
			if filename.endswith('.track'):
				track_name = filename[:-6]
				old_track_path = source + vid_dir + '/' + track_name + '.track'
				new_track_path = source + vid_dir + '/' + track_name + '.mat'
				assert os.path.exists(old_track_path)
				shutil.copyfile(old_track_path, new_track_path)


# cleanUp1(SOURCE_DIR, 'videos_272x192_250_proposals/', 'single_track_videos/')

# cleanUp2('single_track_videos/')
