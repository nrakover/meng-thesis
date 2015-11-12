require 'torch'

local matio = require 'matio'
matio.use_lua_strings = true

dofile('tracker-hmm.lua')
dofile('word-hmm.lua')


SentenceTracker = {}

function SentenceTracker:new(sentence, video_detections_path, video_features_path, video_optflow_path, word_models)
	newObj = {}
	self.__index = self
	setmetatable(newObj, self)

	-- Process video
	newObj:processVideo(video_detections_path, video_features_path, video_optflow_path)

	-- Parse sentence
	newObj:parseSentence(sentence)

	-- Initialize tracker
	newObj.tracker = Tracker:new(newObj.detectionsByFrame, newObj.detectionsOptFlow)

	-- Initialize word trackers
	newObj:buildWordModels(word_models)

	return newObj
end

function SentenceTracker:getBestTrack()
	-- TODO
	self:setPIMemoTable()

end

local function SentenceTracker:PI(k, v)
	-- Use this to index into memo table
	local stateKey = {}
	for i = 1, #v do
		table.insert(stateKey, {v[i]})
	end

	-- Check if value is memoized
	if self.piMemo[k][stateKey] ~= 1 then
		return self.piMemo[k][stateKey]
	end


	local scoreToReturn = nil
	local bestPath = nil

	-- Base Case
	if k == 1 then
		local tracksScore = 0
		for r = 1, #self.roles do
			tracksScore = tracksScore + self.tracker:detectionStrength(k, v[r])
		end

		local wordsScore = 0
		for w = 1, #self.words do
			wordsScore = wordsScore + math.log(self.words[w]:probOfEmission(v[#self.roles + w], k, v[self.wordToRole[w]]))
			wordsScore = wordsScore + math.log(self.words[w]:statePrior(v[#self.roles + w]))
		end

		scoreToReturn = tracksScore + wordsScore
		bestPath = {[1]=v}
	end

	-- Recursive Case
	local tracksScoreA = 0
	for r = 1, #self.roles do
		tracksScoreA = tracksScoreA + self.tracker:detectionStrength(k, v[r])
	end

	local wordsScoreA = 0
	for w = 1, #self.words do
		wordsScoreA = wordsScoreA + math.log(self.words[w]:probOfEmission(v[#self.roles + w], k, v[self.wordToRole[w]]))
	end

	local prevStates = self:possibleStates(k-1)
	local bestTransitionScore = nil
	local bestPathPrefix = nil
	for i,u in ipairs(prevStates) do
		local prevResult = self:PI(k-1, u)
		
		local tracksScoreB = 0
		for r = 1, #self.roles do
			tracksScoreB = tracksScoreB + self.tracker:temporalCoherence(k, u[r], v[r])
		end

		local wordsScoreB = 0
		for w = 1, #self.words do
			wordsScoreB = wordsScoreB + math.log(self.words[w]:probOfTransition(u[#self.roles + w], v[#self.roles + w])
		end

		if bestTransitionScore == nil or prevResult.score+tracksScoreB+wordsScoreB > bestTransitionScore then
			bestTransitionScore = prevResult.score+tracksScoreB+wordsScoreB
			bestPathPrefix = prevResult.path
		end
	end

	scoreToReturn = bestTransitionScore + tracksScoreA + tracksScoreB
	bestPath = {}
	for p = 1, #bestPathPrefix do
		table.insert(bestPath, bestPathPrefix[p])
	end
	table.insert(bestPath, v)
	local result = {score=scoreToReturn, path=bestPath}		-- NEED TO ADD SUPPORT FOR THIS

	-- Memoize
	self.piMemo[k][stateKey] = scoreToReturn

	return scoreToReturn
end

local function SentenceTracker:setPIMemoTable()
	self.piMemo = {}
	for t = 1, #self.detectionsByFrame do
		local dimsTable = {}
		for r = 1, #self.roles do
			table.insert(dimsTable, self.detectionsByFrame[t]:size(1))
		end
		for w = 1, #self.words do
			table.insert(dimsTable, self.words[w].stateTransitions:size(1))
		end

		table.insert(self.piMemo, torch.ones(torch.LongStorage(dimsTable)))
	end
end

local function SentenceTracker:processVideo(video_detections_path, video_features_path, video_optflow_path)
	-- Load the detection proposals
	self.detectionsByFrame = matio.load(video_detections_path , 'detections_by_frame')

	-- Load the optical flow for each frame
	self.detectionsOptFlow = torch.load(video_optflow_path)

	-- Load the neural network features from proposals
	self.detectionFeatures = torch.load(video_features_path)
end

local function SentenceTracker:parseSentence(sentence)
	-- TODO: generalize to more than one word

	self.sentence = sentence 	-- for now, sentence is a single word
	self.roles = {sentence}  	-- for now, sentence is a single word
	self.wordToRole = {[1]=1}	-- for now, sentence is a single word
end

local function SentenceTracker:buildWordModels(word_models)
	-- TODO: generalize to more than one word

	self.words = {}
	self.words[1] = Word:new(word_models[1].emission, word_models[1].transitions, word_models[1].priors, self.detectionsByFrame, self.detectionFeatures)
end

local function SentenceTracker:possibleStates(frameIndx)
	local rolesAssignments = getRoleToDetectionAssignments(#self.roles, self.detectionsByFrame[frameIndx]:size(1))
	local statesAssignments = self:getWordToStateAssignments(#self.words)

	local assignments = {}
	for i = 1, #rolesAssignments do
		local ra = rolesAssignments[i]
		for j = 1, #statesAssignments do
			local sa = statesAssignments[j]
			
			local a = {}
			for r = 1, #ra do
				table.insert(a, ra[r])
			end
			for s = 1, #sa do
				table.insert(a, sa[s])
			end

			table.insert(assignments, a)
		end
	end
	return assignments
end

local function getRoleToDetectionAssignments(numRoles, numDets)
	local assignments = {}
	if numRoles == 1 then
		for d = 1,numDets do
			table.insert(assignments, {[1]=d})
		end
		return assignments
	end

	local subAssignments = getRoleToDetectionAssignments(numRoles-1, numDets)
	for i = 1, #subAssignments do
		local subAssmnt = subAssignments[i]
		for d = 1, numDets do
			local a = {}
			for j = 1, #subAssmnt do
				a[j] = subAssmnt[j]
			end
			table.insert(a, d)
			table.insert(assignments, a)
		end
	end
	return assignments
end

local function SentenceTracker:getWordToStateAssignments(numWords)
	local assignments = {}
	if numWords == 1 then
		for s = 1, self.words[1].stateTransitions:size(1) do
			table.insert(assignments, {[1]=s})
		end
	else
		local subAssignments = self:getWordToStateAssignments(numWords-1)
		for i = 1, #subAssignments do
			local subAssmnt = subAssignments[i]
			for s = 1, self.words[numWords].stateTransitions:size(1) do
				local a = {}
				for j = 1, #subAssmnt do
					a[j] = subAssmnt[j]
				end
				table.insert(a, s)
				table.insert(assignments, a)
			end
		end
	end
	return assignments
end









