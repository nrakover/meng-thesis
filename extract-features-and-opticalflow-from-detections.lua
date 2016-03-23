require 'torch'
require 'nn'
require 'ffmpeg'
require 'liuflow'

torch.setdefaulttensortype('torch.FloatTensor')
require 'loadcaffe'

local matio = require 'matio'
matio.use_lua_strings = true

dofile('/local/nrakover/meng/load-and-process-img.lua')

function distanceTransform( A )
	-- Transformed field
	local B = torch.FloatTensor(A:size())
	
	-- Base case
	B[1][1] = A[1][1]
	-- Leftmost column
	for i = 2, A:size(1) do
		B[i][1] = B[i-1][1] + A[i][1]
	end
	-- Top row
	for j = 2, A:size(2) do
		B[1][j] = B[1][j-1] + A[1][j]
	end
	-- Inner grid
	for i = 2, A:size(1) do
		for j = 2, A:size(2) do
			B[i][j] = B[i-1][j] + B[i][j-1] - B[i-1][j-1] + A[i][j]
		end
	end
	return B
end

function normalizeImage(im)
	local mean_img = torch.FloatTensor(im:size())
	mean_img[{{1},{},{}}] = -123.68
	mean_img[{{2},{},{}}] = -116.779
	mean_img[{{3},{},{}}] = -103.939
	mean_img = mean_img:float()
	return torch.add(im,mean_img):float()
end

local IMG_DIM = 224
local LAYER_TO_EXTRACT = 43
function extractFeatures(img, net)
	local processed_img = processImage(img, IMG_DIM)
	local normd_img = normalizeImage(processed_img)
	net:forward(normd_img)
	local features = net:get(LAYER_TO_EXTRACT).output:clone()
	return torch.squeeze(nn.View(1):forward(features)):double()
end

local net = loadcaffe.load('/local/nrakover/meng/networks/VGG/VGG_ILSVRC_19_layers_deploy.prototxt', '/local/nrakover/meng/networks/VGG/VGG_ILSVRC_19_layers.caffemodel', 'nn')

function extractFeaturesAndOpticalFlow(detectionsByFrame, video_filepath, compute_opticalflow)
	if compute_opticalflow == nil then
		compute_opticalflow = true
	end

	local w = detectionsByFrame.width[1][1]
	local h = detectionsByFrame.height[1][1]
	local frameRate = detectionsByFrame.fps[1][1]
	local duration = detectionsByFrame.length[1][1]
	local vid = ffmpeg.Video{path=video_filepath, height=h, width=w, fps=frameRate, length=duration, silent=true}

	local videoFrames = vid:totensor(1,1,detectionsByFrame.detections:size(1))
	print('Num frames: '..detectionsByFrame.detections:size(1))

	local featuresByFrame = {}
	local opticalflowByFrame = {}

	local prevFrame = nil
	-- Iterate over frames
	for frameIndx = 1,detectionsByFrame.detections:size(1) do
		local frameDetections = detectionsByFrame.detections[frameIndx]
		local frame = videoFrames[frameIndx]

		featuresByFrame[frameIndx] = {}
		if frameIndx ~= 1 and compute_opticalflow then
			local flow_norm, flow_angle, warp, fx, fy = liuflow.infer({prevFrame, frame})

			print('Optical flow computed for frame '..frameIndx)

			opticalflowByFrame[frameIndx] = {flow_x=distanceTransform(torch.squeeze(fx)), flow_y=distanceTransform(torch.squeeze(fy))}
		end
		

		-- for detIndx = detectionsByFrame.person_detector_indices[frameIndx][1], frameDetections:size(1) do
		-- Iterate over detections
		for detIndx = 1,frameDetections:size(1) do
			-- Get image region
			local x_min = math.min(w-2, frameDetections[detIndx][1])
			local y_min = math.min(h-2, frameDetections[detIndx][2])
			local x_max = math.min(w-1, frameDetections[detIndx][3])
			local y_max = math.min(h-1, frameDetections[detIndx][4])

			-- Skip null detections
			if x_min == 0 and x_max == 0 and y_min == 0 and y_max == 0 then
				featuresByFrame[frameIndx][detIndx] = torch.DoubleTensor(4096):fill(0)

			else
				x_min = math.max(1, x_min)
				y_min = math.max(1, y_min)
				if y_max == y_min then y_max = math.min(h-1, y_max + 1) end
				if x_max == x_min then x_max = math.min(w-1, x_max + 1) end

				local frame_region = nil
				if pcall( function() frame_region = image.crop(frame, x_min, y_min, x_max, y_max) end ) then
					-- Compute features
					local frame_region_features = extractFeatures(frame_region, net)
					-- table.insert(featuresByFrame[frameIndx], frame_region_features:clone())
					featuresByFrame[frameIndx][detIndx] = frame_region_features:clone()
				else
					print('\nERROR')
					print(x_min, y_min, x_max, y_max, w, h)
					print(frame:size())
					image.crop(frame, x_min, y_min, x_max, y_max) -- KILLS PROCESS
				end
			end

			-- Show progress on the current frame
			io.write(('  '..(100 * detIndx / frameDetections:size(1)))..'%', '\r'); io.flush();
		end

		prevFrame = frame:clone()

		print('Done with frame '..frameIndx)
	end

	return featuresByFrame, opticalflowByFrame
end

function extractFeaturesGivenFrameAndBounds(bounds, frame)
	-- Get image region
	local x_min = bounds[1]
	local y_min = bounds[2]
	local x_max = bounds[3]
	local y_max = bounds[4]

	if y_max == y_min then y_max = y_max + 1 end
	if x_max == x_min then x_max = x_max + 1 end

	local frame_region = image.crop(frame, x_min, y_min, x_max, y_max)

	-- Compute features
	return extractFeatures(frame_region, net)
end


-- Run on test data
-- local detectionsByFrame = matio.load('script_in/nico2.mat' , 'detections_by_frame')
-- local features, opticalflow = extractFeaturesAndOpticalFlow(detectionsByFrame, 'script_in/nico2.avi')

-- torch.save('script_in/nico2_features.t7', features)
-- torch.save('script_in/nico2_opticalflow.t7', opticalflow)


