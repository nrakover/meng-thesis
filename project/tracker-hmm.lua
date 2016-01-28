require 'torch'
require 'image'

Tracker = {}

function Tracker:new(detections_by_frame, detections_optical_flow, temporal_coherence_exponent)
	temporal_coherence_exponent = temporal_coherence_exponent or 2

	local newObj = {detectionsByFrame=detections_by_frame, detectionsOptFlow=detections_optical_flow, C=temporal_coherence_exponent}
	self.__index = self
	setmetatable(newObj, self)

	newObj:setMemoTables()
	return newObj
end

function Tracker:detectionStrength(frameIndx, detectionIndx)
	-- TODO: insert score if available from object proposal generator
	return 0 -- log(1)
end

function Tracker:temporalCoherence(frameIndx, prevDetectionIndx, detectionIndx)
	-- Base case:
	if frameIndx == 1 then return 0 end -- log(1)

	-- Check if value is memoized
	if self.memo[frameIndx][prevDetectionIndx][detectionIndx] ~= 1 then
		return self.memo[frameIndx][prevDetectionIndx][detectionIndx]
	end

	-- Else use optical flow to compute distance:

	-- Get detection bounds for previous frame detection
	local prev_x_min = self.detectionsByFrame[frameIndx-1][prevDetectionIndx][1]
	local prev_y_min = self.detectionsByFrame[frameIndx-1][prevDetectionIndx][2]
	local prev_x_max = self.detectionsByFrame[frameIndx-1][prevDetectionIndx][3]
	local prev_y_max = self.detectionsByFrame[frameIndx-1][prevDetectionIndx][4]

	if prev_y_max == prev_y_min then prev_y_max = prev_y_max + 1 end
	if prev_x_max == prev_x_min then prev_x_max = prev_x_max + 1 end

	-- Previous detection's center
	local prev_center = torch.Tensor( {(prev_x_max+prev_x_min)/2, (prev_y_max+prev_y_min)/2} )

	-- Get average flow from region
	local avg_flow_x = self:extractAvgFlowFromDistanceTransform(self.detectionsOptFlow[frameIndx].flow_x, prev_x_min, prev_y_min, prev_x_max, prev_y_max)
	local avg_flow_y = self:extractAvgFlowFromDistanceTransform(self.detectionsOptFlow[frameIndx].flow_y, prev_x_min, prev_y_min, prev_x_max, prev_y_max)
	local avg_flow = torch.Tensor( { avg_flow_x, avg_flow_y } )

	-- Project previous detection's center
	local projected_center = prev_center + avg_flow

	-- Get detection bounds for current frame
	local x_min = self.detectionsByFrame[frameIndx][detectionIndx][1]
	local y_min = self.detectionsByFrame[frameIndx][detectionIndx][2]
	local x_max = self.detectionsByFrame[frameIndx][detectionIndx][3]
	local y_max = self.detectionsByFrame[frameIndx][detectionIndx][4]

	if y_max == y_min then y_max = y_max + 1 end
	if x_max == x_min then x_max = x_max + 1 end

	-- Current detection's center
	local current_center = torch.Tensor( {(x_max+x_min)/2, (y_max+y_min)/2} )


	-- Negative Euclidean distance between previous detection's center and backprojected center
	local d = torch.dist(projected_center, current_center)
	-- Normalize the distance into [0,1]
	local max_d = torch.dist( torch.Tensor({1,1}), torch.Tensor({self.detectionsOptFlow[frameIndx].flow_x:size(1), self.detectionsOptFlow[frameIndx].flow_x:size(2)}) )
	local score = math.log( 1 - (d / max_d) ) * self.C

	-- Memoize
	self.memo[frameIndx][prevDetectionIndx][detectionIndx] = score
	return score
end

function Tracker:setMemoTables()
	self.memo = {}
	for t = 2, #self.detectionsByFrame do
		self.memo[t] = torch.ones(#self.detectionsByFrame[t-1], #self.detectionsByFrame[t])
	end
end

function Tracker:extractAvgFlowFromDistanceTransform(flow_field, x_min, y_min, x_max, y_max)
	local flow_sum = flow_field[y_max][x_max] - flow_field[y_min][x_max] - flow_field[y_max][x_min] + flow_field[y_min][x_min]
	local area = (y_max-y_min) * (x_max-x_min)
	return flow_sum / area
end




