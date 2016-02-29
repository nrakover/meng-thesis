require 'image'
require 'ffmpeg'


local matio = require 'matio'
matio.use_lua_strings = true

local function addBBoxToFrame(bbox, frame)
	if bbox[1] == 0 or bbox[2] == 0 or bbox[3] == 0 or bbox[4] == 0 then return frame end

	local x_min = bbox[1]
	local y_min = bbox[2]
	local x_max = bbox[3]
	local y_max = bbox[4]

	-- Draw left vertical line
	frame[{{1},{y_min,y_max},{x_min}}] = 1
	frame[{{2},{y_min,y_max},{x_min}}] = 0
	frame[{{3},{y_min,y_max},{x_min}}] = 0

	-- Draw right vertical line
	frame[{{1},{y_min,y_max},{x_max}}] = 1
	frame[{{2},{y_min,y_max},{x_max}}] = 0
	frame[{{3},{y_min,y_max},{x_max}}] = 0

	-- Draw top horizontal line
	frame[{{1},{y_min},{x_min,x_max}}] = 1
	frame[{{2},{y_min},{x_min,x_max}}] = 0
	frame[{{3},{y_min},{x_min,x_max}}] = 0

	-- Draw bottom horizontal line
	frame[{{1},{y_max},{x_min,x_max}}] = 1
	frame[{{2},{y_max},{x_min,x_max}}] = 0
	frame[{{3},{y_max},{x_min,x_max}}] = 0

	return frame
end

function visualizeProposals(detections_path, video_path)
	local detectionsByFrame = matio.load(detections_path , 'detections_by_frame')
	
	local w = detectionsByFrame.width[1][1]
	local h = detectionsByFrame.height[1][1]
	local frameRate = detectionsByFrame.fps[1][1]
	local duration = detectionsByFrame.length[1][1]
	local vid = ffmpeg.Video{path=video_path, height=h, width=w, fps=frameRate, length=duration, silent=true}

	local numFrames = detectionsByFrame.detections:size(1)
	local videoFrames = vid:totensor(1,1,numFrames)
	
	for fIndx = 1, numFrames do
		local frame = videoFrames[fIndx]
		for detIndx = 1, detectionsByFrame.detections:size(2) do
			local bbox = detectionsByFrame.detections[fIndx][detIndx]
			frame = addBBoxToFrame(bbox, frame)
		end
		image.display(frame)
	end
end

-- visualizeProposals('/local/nrakover/meng/datasets/video-corpus/single_track_videos/MVI_0874a/detections.mat', '/local/nrakover/meng/datasets/video-corpus/single_track_videos/MVI_0874/video.avi')

