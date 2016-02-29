require 'image'
require 'ffmpeg'


local matio = require 'matio'
matio.use_lua_strings = true

local function addBBoxToFrame(bbox, frame, max_width, max_height)
	if bbox[1] == 0 or bbox[2] == 0 or bbox[3] == 0 or bbox[4] == 0 then return frame end

	local x_min = bbox[1]
	local y_min = bbox[2]
	local x_max = math.min( max_width-1, bbox[3] )
	local y_max = math.min( max_height-1, bbox[4] )

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

function visualizeTrackMAT(track_path, detections_path, video_path)
	local track = matio.load(track_path)

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
		for detIndx = 1, track['t'..fIndx]:size(1) do
			local bbox = track['t'..fIndx][detIndx]
			frame = addBBoxToFrame(bbox, frame, w, h)
		end
		image.display(frame)
	end
end

