require 'torch'
require 'svm'

dofile('../data-to-svmlight.lua' );

Word = {}

function Word:new(emission_models, state_transitions, state_priors, detections_by_frame, detection_features)
	newObj = {emissionModels=emission_models, stateTransitions=state_transitions, statePriors=state_priors, detectionsByFrame=detections_by_frame, detectionFeatures=detection_features}
	self.__index = self
	setmetatable(newObj, self)

	newObj:setMemoTables()
	return newObj
end

function Word:probOfEmission(state, frameIndx, detectionIndx)
	-- return memoized value if present
	if self.memo[frameIndx][state][detectionIndx] ~= -1 then
		return self.memo[frameIndx][state][detectionIndx]
	end

	-- else compute value
	local features_for_frame_detections = self.detectionFeatures[frameIndx]
	local formatted_features = t7ToSvmlight(features_for_frame_detections, torch.ones(features_for_frame_detections:size(1)))

	local labels,accuracy,prob = liblinear.predict(formatted_features, self.emissionModels[state], '-b 1 -q')
	
	-- memoize
	local positive_class_indx = 1
	if self.emissionModels[state].label[2] == 1 then positive_class_indx = 2 end
	for d = 1,prob:size(1) do
		self.memo[frameIndx][state][d] = prob[d][positive_class_indx]
	end

	return self.memo[frameIndx][state][detectionIndx]
end

function Word:probOfTransition(prevState, newState)
	return self.stateTransitions[prevState][newState]
end

function Word:statePrior(state)
	return self.statePriors[state]
end

local function Word:setMemoTables()
	self.memo = {}
	local nStates = self.stateTransitions:size(1)
	for t = 1,#self.detectionsByFrame do
		table.insert(self.memo, -torch.ones(nStates, #self.detectionsByFrame[t]))
	end
end

	 
