require 'torch'
require 'image'

Tracker = {}

function Tracker:new(detections_by_frame, detections_optical_flow)
	newObj = {detectionsByFrame=detections_by_frame, detectionsOptFlow=detections_optical_flow}
	self.__index = self
	setmetatable(newObj, self)

	newObj:setMemoTables()
	return newObj
end

function Tracker:detectionStrength(frameIndx, detectionIndx)
	-- TODO: consider better metric
	return 1/self.detectionsByFrame[frameIndx]:size(1)
end

function Tracker:temporalCoherence(frameIndx, prevDetectionIndx, detectionIndx)
	-- Base case:
	if frameIndx == 1 then return 0 end

	-- Check if value is memoized
	if self.memo[frameIndx][prevDetectionIndx][detectionIndx] ~= -1 then
		return self.memo[frameIndx][prevDetectionIndx][detectionIndx]
	end

	-- Else use optical flow to compute distance:

	-- Get detection bounds
	local x_min = self.detectionsByFrame[frameIndx][detectionIndx][1]
	local y_min = self.detectionsByFrame[frameIndx][detectionIndx][2]
	local x_max = self.detectionsByFrame[frameIndx][detectionIndx][3]
	local y_max = self.detectionsByFrame[frameIndx][detectionIndx][4]

	if y_max == y_min then y_max = y_max + 1 end
	if x_max == x_min then x_max = x_max + 1 end

	-- Get optical flow
	local flow_x_region = image.crop(self.detectionsOptFlow[frameIndx].flow_x, x_min, y_min, x_max, y_max)
	local flow_y_region = image.crop(self.detectionsOptFlow[frameIndx].flow_y, x_min, y_min, x_max, y_max)

	-- Compute summary of flow field
	-- TODO: use 2D-gaussian for weighted average
	local avg_flow = torch.Tensor( {torch.mean(flow_x_region), torch.mean(flow_y_region)} )

	-- Project current detection's center
	local projected_center = torch.Tensor( {(x_max+x_min)/2, (y_max+y_min)/2} ) + avg_flow


	-- Get detection bounds for previous frame detection
	local prev_x_min = self.detectionsByFrame[frameIndx-1][prevDetectionIndx][1]
	local prev_y_min = self.detectionsByFrame[frameIndx-1][prevDetectionIndx][2]
	local prev_x_max = self.detectionsByFrame[frameIndx-1][prevDetectionIndx][3]
	local prev_y_max = self.detectionsByFrame[frameIndx-1][prevDetectionIndx][4]

	if prev_y_max == prev_y_min then prev_y_max = prev_y_max + 1 end
	if prev_x_max == prev_x_min then prev_x_max = prev_x_max + 1 end

	-- Previous detection's center
	local prev_center = torch.Tensor( {(prev_x_max+prev_x_min)/2, (prev_y_max+prev_y_min)/2} )


	-- Return negative Euclidean distance between previous detection's center and backprojected center
	local d = -torch.dist(projected_center, prev_center)
	-- Memoize
	self.memo[frameIndx][prevDetectionIndx][detectionIndx] = d
	return d
end

local function Tracker:setMemoTables()
	self.memo = {}
	for t = 2,#self.detectionsByFrame do
		table.insert( self.memo, -torch.ones(self.detectionsByFrame[t-1]:size(1), self.detectionsByFrame[t]:size(1)) )
	end
end
