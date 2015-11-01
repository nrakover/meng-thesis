require 'torch'

Tracker = {}

function Tracker:new(detections_by_frame, detection_scores)
	newObj = {detectionsByFrame=detections_by_frame , detectionScores=detection_scores}
	self.__index = self
	return setmetatable(newObj, self)
end

function Tracker:detectionStrength(frameIndx, detectionIndx)
	return self.detectionScores[frameIndx][detectionIndx]
end

function Tracker:temporalCoherence(frameIndx, prevDetecetionIndx, detectionIndx)
	-- Base case:
	if frameIndx == 1 then return 0 end

	-- Use optical flow to compute distance
	
end