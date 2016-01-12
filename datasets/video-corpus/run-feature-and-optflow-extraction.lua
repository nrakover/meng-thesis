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

function extractFeaturesAndOptflowFromAllVideos(source_dir)
	local all_vid_dirs = dirLookup(source_dir)

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


extractFeaturesAndOptflowFromAllVideos('preprocessed-videos')