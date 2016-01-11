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

	-- local formatted_features = t7ToSvmlight({[1]=stacked_features_for_detections}, torch.ones(1))

	-- local labels,accuracy,prob = liblinear.predict(formatted_features, self.emissionModels[state], '-b 1 -q')
	local prob = self.emissionModels[state]:forward(stacked_features_for_detections)
	
	-- memoize
	self.memo[key] = prob[1]
	-- local positive_class_indx = 1
	-- if self.emissionModels[state].label[2] == 1 then positive_class_indx = 2 end
	-- self.memo[key] = prob[1][positive_class_indx]

	return self.memo[key]
end

function Word:probOfTransition(prevState, newState)
	return self.stateTransitions[prevState][newState]
end

function Word:statePrior(state)
	return self.statePriors[state]
end

function Word:setMemoTables()
	self.memo = {}
end

function Word:getKey(state, frame, detections)
	local key = (''..state)..(':'..frame)..':'
	for i = 1, #detections do
		key = (key..detections[i])..'_'
	end
	return key
end

function Word:extractFeatures( frameIndx, detections )
	local stacked_features_for_detections = self.detectionFeatures[frameIndx][detections[1]]:clone()
	if #detections > 1 then
		local avg_flow_vec = self:extractAvgFlowFromDistanceTransform(frameIndx, detections[1])
		stacked_features_for_detections = torch.cat(stacked_features_for_detections, avg_flow_vec, 1)

		local center = self:getNormalizedDetectionCenter(frameIndx, detections[1])
		stacked_features_for_detections = torch.cat(stacked_features_for_detections, center, 1)
	end

	for i = 2, #detections do
		-- VGG features
		stacked_features_for_detections = torch.cat(stacked_features_for_detections, self.detectionFeatures[frameIndx][detections[i]]:clone(), 1)

		-- Average optical flow vector
		local avg_flow_vec = self:extractAvgFlowFromDistanceTransform(frameIndx, detections[i])
		stacked_features_for_detections = torch.cat(stacked_features_for_detections, avg_flow_vec, 1)

		-- Detection center, with coordinates normalized to [0,1]
		local center = self:getNormalizedDetectionCenter(frameIndx, detections[i])
		stacked_features_for_detections = torch.cat(stacked_features_for_detections, center, 1)
	end
	return torch.squeeze(stacked_features_for_detections):double()
end

function Word:extractAvgFlowFromDistanceTransform( frameIndx, detectionIndx )
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
	local frame_width = self.framesOptFlow[frameIndx].flow_x:size(2)
	local frame_height = self.framesOptFlow[frameIndx].flow_x:size(1)

	return torch.Tensor({((x_max + x_min)/2)/frame_width, ((y_max + y_min)/2)/frame_height})
end

	 
