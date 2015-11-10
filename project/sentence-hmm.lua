
dofile('../torch-libs/liuflow/init.lua')

SentenceTracker = {}

function SentenceTracker:new(sentence, video, word_emission_models)
	newObj = {}
	self.__index = self
	setmetatable(newObj, self)

	-- Process video
	newObj:processVideo(video)

	-- Parse sentence
	newObj:parseSentence(sentence)

	-- Initialize word trackers
	newObj:buildWordTrackers(word_emission_models)

	return newObj
end

local function SentenceTracker:processVideo(video)
	-- TODO: implement

	-- compute the detection proposals

	-- extract optical flow from proposals

	-- extract neural network features from proposals

end

local function SentenceTracker:parseSentence()
	-- TODO: implement
end

local function SentenceTracker:buildWordTrackers(word_emission_models)
	-- TODO: implement
end

local function SentenceTracker:possibleStates(frameIndx)
	-- TODO: generalize to more than one word

end

