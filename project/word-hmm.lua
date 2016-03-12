require 'torch'
require 'nn'
-- require 'svm'

-- dofile('/local/nrakover/meng/data-to-svmlight.lua');

Word = {}

function Word:new(emission_models, state_transitions, state_priors, detections_by_frame, detection_features, frames_optical_flow)
	local newObj = {emissionModels=emission_models, stateTransitions=state_transitions, statePriors=state_priors, detectionsByFrame=detections_by_frame, detectionFeatures=detection_features, framesOptFlow=frames_optical_flow}
	self.__index = self
	setmetatable(newObj, self)

	newObj:setMemoTables()
	return newObj
end

function Word:probOfEmission(state, frameIndx, detections)
	local key = self:getKey(state, frameIndx, detections)

	-- return memoized value if present
	if self.memo[key] ~= nil then
		return self.memo[key]
	end

	-- else compute value
	local stacked_features_for_detections = self:extractFeatures( frameIndx, detections )
	local prob = self.emissionModels[state]:forward(stacked_features_for_detections)
	
	-- memoize
	self.memo[key] = prob[1]

	return self.memo[key]
end

function Word:probOfTransition(prevState, newState)
	return self.stateTransitions[prevState][newState]
end

function Word:statePrior(state)
	-- Constrain starting state to be first state
	if state == 1 then
		return 1
	else
		return 0
	end
end

function Word:stateTerminalDistribution(state)
	-- Constrain terminal state to be the last state 
	if state == self.stateTransitions:size(1) then
		return 1
	else
		return 0
	end
end

function Word:setMemoTables()
	self.memo = {}
	self.featureExtractionMemo = {}
end

function Word:getKey(state, frame, detections)
	local key = (''..state)..(':'..frame)..':'
	for i = 1, #detections do
		key = (key..detections[i])..'_'
	end
	return key
end

function Word:extractFeatures( frameIndx, detections )
	local key = self:getKey(0,frameIndx,detections)

	-- return memoized value if present
	if self.featureExtractionMemo[key] ~= nil then
		return self.featureExtractionMemo[key]
	end

	local stacked_features_for_detections = nil 

	-- #################################### WITH VGG ####################################
	stacked_features_for_detections = self.detectionFeatures[frameIndx][detections[1]]:clone():double() * 0.1
	if #detections == 2 then -- ########################### TEMPORARY ############################
		stacked_features_for_detections:mul(0.001)

		local avg_flow_vec1 = self:extractAvgFlowFromDistanceTransform(frameIndx, detections[1])
		stacked_features_for_detections = torch.cat(stacked_features_for_detections, avg_flow_vec1, 1)

		local center1 = self:getNormalizedDetectionCenter(frameIndx, detections[1])
		-- stacked_features_for_detections = torch.cat(stacked_features_for_detections, center1, 1)

		stacked_features_for_detections = torch.cat(stacked_features_for_detections, self.detectionFeatures[frameIndx][detections[2]]:clone():double() * 0.0001, 1)

		local avg_flow_vec2 = self:extractAvgFlowFromDistanceTransform(frameIndx, detections[2])
		stacked_features_for_detections = torch.cat(stacked_features_for_detections, avg_flow_vec2, 1)

		local center2 = self:getNormalizedDetectionCenter(frameIndx, detections[2])
		-- stacked_features_for_detections = torch.cat(stacked_features_for_detections, center2, 1)

		-- ########################### TEMPORARY ############################
		stacked_features_for_detections = torch.cat(stacked_features_for_detections, torch.DoubleTensor({torch.dist(center2, center1)}), 1)
		stacked_features_for_detections = torch.cat(stacked_features_for_detections, center2 - center1, 1)
		stacked_features_for_detections = torch.cat(stacked_features_for_detections, avg_flow_vec2 - avg_flow_vec1, 1)

		local horizontal_dist = self:getNormalizedClosestSideDistance( frameIndx, detections[1], detections[2] )
		stacked_features_for_detections = torch.cat(stacked_features_for_detections, torch.DoubleTensor({horizontal_dist}), 1)
		-- ########################### TEMPORARY ############################
		stacked_features_for_detections:mul(1000)
	end
	-- #################################### WITH VGG ####################################

	-- -- #################################### NO VGG ####################################
	-- if #detections == 1 then
	-- 	stacked_features_for_detections = self.detectionFeatures[frameIndx][detections[1]]:clone():double()
	-- elseif #detections == 2 then -- ########################### TEMPORARY ############################
	-- 	local avg_flow_vec1 = self:extractAvgFlowFromDistanceTransform(frameIndx, detections[1])
	-- 	-- stacked_features_for_detections = torch.cat(stacked_features_for_detections, avg_flow_vec1, 1)
	-- 	stacked_features_for_detections = avg_flow_vec1

	-- 	local center1 = self:getNormalizedDetectionCenter(frameIndx, detections[1])
	-- 	-- stacked_features_for_detections = torch.cat(stacked_features_for_detections, center1, 1)

	-- 	-- stacked_features_for_detections = torch.cat(stacked_features_for_detections, self.detectionFeatures[frameIndx][detections[2]]:clone():double(), 1)

	-- 	local avg_flow_vec2 = self:extractAvgFlowFromDistanceTransform(frameIndx, detections[2])
	-- 	stacked_features_for_detections = torch.cat(stacked_features_for_detections, avg_flow_vec2, 1)

	-- 	local center2 = self:getNormalizedDetectionCenter(frameIndx, detections[2])
	-- 	-- stacked_features_for_detections = torch.cat(stacked_features_for_detections, center2, 1)

	-- 	-- ########################### TEMPORARY ############################
	-- 	stacked_features_for_detections = torch.cat(stacked_features_for_detections, torch.DoubleTensor({torch.dist(center2, center1)}), 1)
	-- 	stacked_features_for_detections = torch.cat(stacked_features_for_detections, center2 - center1, 1)
	-- 	stacked_features_for_detections = torch.cat(stacked_features_for_detections, avg_flow_vec2 - avg_flow_vec1, 1)

	-- 	local horizontal_dist = self:getNormalizedClosestSideDistance( frameIndx, detections[1], detections[2] )
	-- 	stacked_features_for_detections = torch.cat(stacked_features_for_detections, torch.DoubleTensor({horizontal_dist}), 1)
	-- 	-- ########################### TEMPORARY ############################
	-- 	stacked_features_for_detections:mul(10)
	-- end
	-- -- #################################### NO VGG ####################################

	-- for i = 2, #detections do
	-- 	-- VGG features
	-- 	stacked_features_for_detections = torch.cat(stacked_features_for_detections, self.detectionFeatures[frameIndx][detections[i]]:clone():double(), 1)

	-- 	-- Average optical flow vector
	-- 	local avg_flow_vec = self:extractAvgFlowFromDistanceTransform(frameIndx, detections[i])
	-- 	stacked_features_for_detections = torch.cat(stacked_features_for_detections, avg_flow_vec, 1)

	-- 	-- Detection center, with coordinates normalized to [0,1]
	-- 	local center = self:getNormalizedDetectionCenter(frameIndx, detections[i])
	-- 	stacked_features_for_detections = torch.cat(stacked_features_for_detections, center, 1)
	-- end

	
	self.featureExtractionMemo[key] = torch.squeeze(stacked_features_for_detections):double()
	return self.featureExtractionMemo[key]
end

function Word:extractAvgFlowFromDistanceTransform( frameIndx, detectionIndx )

	-- No motion information for first frame
	if frameIndx == 1 then
		return torch.zeros(2)
	end

	-- Get detection bounds
	local x_min = self.detectionsByFrame[frameIndx][detectionIndx][1]
	local y_min = self.detectionsByFrame[frameIndx][detectionIndx][2]
	local x_max = self.detectionsByFrame[frameIndx][detectionIndx][3]
	local y_max = self.detectionsByFrame[frameIndx][detectionIndx][4]

	-- Correct for degenerate bounding boxes
	if y_max == y_min then y_max = y_max + 1 end
	if x_max == x_min then x_max = x_max + 1 end

	-- Compute average flow
	local flow_field_x = self.framesOptFlow[frameIndx].flow_x
	x_max = math.min(flow_field_x:size(2)-1, x_max)
	y_max = math.min(flow_field_x:size(1)-1, y_max)
	local flow_sum_x = flow_field_x[y_max][x_max] - flow_field_x[y_min][x_max] - flow_field_x[y_max][x_min] + flow_field_x[y_min][x_min]

	local flow_field_y = self.framesOptFlow[frameIndx].flow_y
	local flow_sum_y = flow_field_y[y_max][x_max] - flow_field_y[y_min][x_max] - flow_field_y[y_max][x_min] + flow_field_y[y_min][x_min]
	
	local area = (y_max-y_min) * (x_max-x_min)
	return  torch.Tensor({ flow_sum_x / area , flow_sum_y / area })
end

function Word:getNormalizedDetectionCenter( frameIndx, detectionIndx )
	-- Get detection bounds
	local x_min = self.detectionsByFrame[frameIndx][detectionIndx][1]
	local y_min = self.detectionsByFrame[frameIndx][detectionIndx][2]
	local x_max = self.detectionsByFrame[frameIndx][detectionIndx][3]
	local y_max = self.detectionsByFrame[frameIndx][detectionIndx][4]

	-- Correct for degenerate bounding boxes
	if y_max == y_min then y_max = y_max + 1 end
	if x_max == x_min then x_max = x_max + 1 end

	-- Get frame dimensions
	local frame_width = self.framesOptFlow[2].flow_x:size(2)
	local frame_height = self.framesOptFlow[2].flow_x:size(1)

	return torch.Tensor({((x_max + x_min)/2)/frame_width, ((y_max + y_min)/2)/frame_height})
end

function Word:getNormalizedClosestSideDistance( frameIndx, detectionIndx1, detectionIndx2 )
	-- Get detection bounds 1
	local x_min1 = self.detectionsByFrame[frameIndx][detectionIndx1][1]
	local x_max1 = self.detectionsByFrame[frameIndx][detectionIndx1][3]

	-- Correct for degenerate bounding boxes
	if x_max1 == x_min1 then x_max1 = x_max1 + 1 end


	-- Get detection bounds 2
	local x_min2 = self.detectionsByFrame[frameIndx][detectionIndx2][1]
	local x_max2 = self.detectionsByFrame[frameIndx][detectionIndx2][3]

	-- Correct for degenerate bounding boxes
	if x_max2 == x_min2 then x_max2 = x_max2 + 1 end


	-- Compute shortest horizontal distance 
	if (x_min2 >= x_min1 and x_min2 <= x_max1) or (x_max2 >= x_min1 and x_max2 <= x_max1) or (x_min2 <= x_min1 and x_max2 >= x_max1) then return 0 end
	-- Get frame dimensions
	local frame_width = self.framesOptFlow[2].flow_x:size(2)
	if x_min2 > x_max1 then
		return (x_min2 - x_max1) / frame_width
	else
		return (x_min1 - x_max2) / frame_width
	end
end

	 
