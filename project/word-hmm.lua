require 'torch'
require 'svm'

dofile('/local/nrakover/meng/data-to-svmlight.lua');

Word = {}

function Word:new(emission_models, state_transitions, state_priors, detections_by_frame, detection_features)
	local newObj = {emissionModels=emission_models, stateTransitions=state_transitions, statePriors=state_priors, detectionsByFrame=detections_by_frame, detectionFeatures=detection_features}
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
	local stacked_features_for_detections = self.detectionFeatures[frameIndx][detections[1]]:clone()
	for i = 2, #detections do
		stacked_features_for_detections = torch.cat(stacked_features_for_detections, self.detectionFeatures[frameIndx][detections[i]]:clone(), 1)
	end

	local formatted_features = t7ToSvmlight({[1]=stacked_features_for_detections}, torch.ones(1))

	local labels,accuracy,prob = liblinear.predict(formatted_features, self.emissionModels[state], '-b 1 -q')
	
	-- memoize
	local positive_class_indx = 1
	if self.emissionModels[state].label[2] == 1 then positive_class_indx = 2 end
	self.memo[key] = prob[1][positive_class_indx]

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

	 
