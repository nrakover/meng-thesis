import os


SOURCE_DIR = '20121214-new3-corpus/'


def generateDirectoryForEachVideo(source, destination):
	source_dir_contents = os.listdir(source)

	for filename in source_dir_contents:
		if filename.endswith('.mov'):
			vid_name = filename[:-4]

			new_dir_path = destination + vid_name
			if not os.path.exists(new_dir_path):
				os.makedirs(new_dir_path)

generateDirectoryForEachVideo(SOURCE_DIR, 'videos_272x192_all-frames_75-proposals/')
