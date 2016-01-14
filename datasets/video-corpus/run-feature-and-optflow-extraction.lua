require 'torch'

local matio = require 'matio'
matio.use_lua_strings = true

dofile('/local/nrakover/meng/extract-features-and-opticalflow-from-detections.lua')

local function dirLookup(dir)
	local p = io.popen('find "'..dir..'" -type d')  --Open directory look for directories, save data in p. By giving '-type d' as parameter, it returns all directories.     
	local dir_names = {}
	for potential_dir in p:lines() do                      --Loop through all dirs
		if string.find(potential_dir, '/MVI_') ~= nil then
			table.insert(dir_names, potential_dir)
		end
	end
	return dir_names
end

function extractFeaturesAndOptflowFromAllVideos(source_dir, videos_to_process)
	local all_vid_dirs = {}
	if videos_to_process == nil then
		all_vid_dirs = dirLookup(source_dir)
	else
		for i,vid in ipairs(videos_to_process) do
			all_vid_dirs[i] = source_dir..'/'..vid
		end
	end

	for i = 1, #all_vid_dirs do
		local vid_dir = all_vid_dirs[i]
		local vid_path = vid_dir..'/video.avi'
		local detections_path = vid_dir..'/detections.mat'

		local detections = matio.load(detections_path, 'detections_by_frame')
		local features, opticalflow = extractFeaturesAndOpticalFlow(detections, vid_path)

		torch.save(vid_dir..'/features.t7', features)
		torch.save(vid_dir..'/opticalflow.t7', opticalflow)

		-- Display progress
		print('\t\t==> '..(100*i/#all_vid_dirs))
	end
end



local vids_to_process = {
	'MVI_0822',
	'MVI_0823',
	'MVI_0824',
	'MVI_0825',
	'MVI_0826',
	'MVI_0827',
	'MVI_0835',
	'MVI_0836',
	'MVI_0837',
	'MVI_0848',
	'MVI_0850',
	'MVI_0855',
	'MVI_0856',
	'MVI_0857',
	'MVI_0858',
	'MVI_0859',
	'MVI_0868',
	'MVI_0869',
	'MVI_0870',
	'MVI_0882',
	'MVI_0884',
	'MVI_0885',
	'MVI_0886',
	'MVI_0887',
	'MVI_0888',
	'MVI_0889',
	'MVI_0890',
	'MVI_0891',
	'MVI_0898',
	'MVI_0900',
	'MVI_0913',
	'MVI_0915'
}

extractFeaturesAndOptflowFromAllVideos('videos_272x192_all-frames_75-proposals', vids_to_process)




