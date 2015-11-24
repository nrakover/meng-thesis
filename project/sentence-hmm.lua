require 'torch'

local matio = require 'matio'
matio.use_lua_strings = true

dofile('/afs/csail.mit.edu/u/n/nrakover/meng/project/tracker-hmm.lua')
dofile('/afs/csail.mit.edu/u/n/nrakover/meng/project/word-hmm.lua')


SentenceTracker = {}

function SentenceTracker:new(sentence, video_detections_path, video_features_path, video_optflow_path, word_models)
	local newObj = {}
	self.__index = self
	newObj = setmetatable(newObj, self)

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
	local path = self:getBestPath()

	-- TODO: generalize to more that one word
	local track = {}
	for frameIndx = 1, #path do
		local state = path[frameIndx]
		local detIndx = state[1]
		track[frameIndx] = self.detectionsByFrame[frameIndx][detIndx]:clone()
	end
	return track
end

function SentenceTracker:getBestPath()
	self:setPIMemoTable()

	local numFrames = self.detectionsByFrame:size(1)
	local v = self:startNode()
	local bestScore = nil
	local bestPath = nil
	while v ~= nil do
		local piResult = self:PI(numFrames, v)
		if bestScore == nil or piResult.score > bestScore then
			bestScore = piResult.score
			bestPath = piResult.path
		end
		v = self:nextNode(v)
	end

	print('==> FINISHED')
	return bestPath, bestScore
end

function SentenceTracker:PI(k, v)
	-- Use this to index into memo table
	local key = getKey(k,v)

	-- Check if value is memoized
	if self.piMemo[key] ~= nil then
		return self.piMemo[key]
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
	
	-- Recursive Case
	else
		local tracksScoreA = 0
		for r = 1, #self.roles do
			tracksScoreA = tracksScoreA + self.tracker:detectionStrength(k, v[r])
		end

		local wordsScoreA = 0
		for w = 1, #self.words do
			wordsScoreA = wordsScoreA + math.log(self.words[w]:probOfEmission(v[#self.roles + w], k, v[self.wordToRole[w]]))
		end

		local u = self:startNode()
		local bestTransitionScore = nil
		local bestPathPrefix = nil
		while u ~= nil do			
			local prevResult = self:PI(k-1, u)
			
			local tracksScoreB = 0
			for r = 1, #self.roles do
				tracksScoreB = tracksScoreB + self.tracker:temporalCoherence(k, u[r], v[r])
			end

			local wordsScoreB = 0
			for w = 1, #self.words do
				wordsScoreB = wordsScoreB + math.log(self.words[w]:probOfTransition(u[#self.roles + w], v[#self.roles + w]))
			end

			if bestTransitionScore == nil or prevResult.score+tracksScoreB+wordsScoreB > bestTransitionScore then
				bestTransitionScore = prevResult.score+tracksScoreB+wordsScoreB
				bestPathPrefix = prevResult.path
			end
			u = self:nextNode(u)
		end

		scoreToReturn = bestTransitionScore + tracksScoreA + wordsScoreA
		bestPath = {}
		for p = 1, #bestPathPrefix do
			table.insert(bestPath, bestPathPrefix[p])
		end
		table.insert(bestPath, v)
	end

	local result = {score=scoreToReturn, path=bestPath}

	-- Memoize
	self.piMemo[key] = result

	print('Computed: '..key)

	return result
end

function SentenceTracker:setPIMemoTable()
	self.piMemo = {}
	self.statesMemo = {}
	-- for t = 1, #self.detectionsByFrame do
	-- 	local dimsTable = {}
	-- 	for r = 1, #self.roles do
	-- 		table.insert(dimsTable, self.detectionsByFrame[t]:size(1))
	-- 	end
	-- 	for w = 1, #self.words do
	-- 		table.insert(dimsTable, self.words[w].stateTransitions:size(1))
	-- 	end

	-- 	table.insert(self.piMemo, torch.ones(torch.LongStorage(dimsTable)))
	-- end
end

function SentenceTracker:processVideo(video_detections_path, video_features_path, video_optflow_path)
	-- Load the detection proposals
	self.detectionsByFrame = matio.load(video_detections_path , 'detections_by_frame').detections

	-- Load the optical flow for each frame
	self.detectionsOptFlow = torch.load(video_optflow_path)

	-- Load the neural network features from proposals
	self.detectionFeatures = torch.load(video_features_path)
end

function SentenceTracker:parseSentence(sentence)
	-- TODO: generalize to more than one word

	self.sentence = sentence 	-- for now, sentence is a single word
	self.roles = {sentence}  	-- for now, sentence is a single word
	self.wordToRole = {[1]=1}	-- for now, sentence is a single word
end

function SentenceTracker:buildWordModels(word_models)
	-- TODO: generalize to more than one word

	self.words = {}
	self.words[1] = Word:new(word_models[1].emissions, word_models[1].transitions, word_models[1].priors, self.detectionsByFrame, self.detectionFeatures)
end

function SentenceTracker:startNode()
	local node = {}
	for i = 1, #self.roles + #self.words do
		node[i] = 1
	end
	return node
end


function SentenceTracker:nextNode( node )
	local nextNode = {}

	local carry = 1
	for i = #node, 1, -1 do
		if node[i] == self:maxValueAt(i) and carry == 1 then
			if i == 1 then return nil end
			nextNode[i] = 1
		else
			nextNode[i] = node[i] + carry
			carry = 0
		end
	end
	return nextNode
end

function SentenceTracker:maxValueAt( i )
	if i <= #self.roles then
		return self.detectionsByFrame:size(2)
	else
		return self.words[i - #self.roles].stateTransitions:size(1)
	end
end

function getKey(k, v)
	local key = (''..k)..':'
	for i = 1, #v do
		key = (key..v[i])..'_'
	end
	return key
end

-- function SentenceTracker:possibleStates(frameIndx)
-- 	if self.statesMemo[frameIndx] ~= nil then
-- 		return self.statesMemo[frameIndx]
-- 	end

-- 	local rolesAssignments = getRoleToDetectionAssignments(#self.roles, self.detectionsByFrame[frameIndx]:size(1))
-- 	local statesAssignments = self:getWordToStateAssignments(#self.words)

-- 	local assignments = {}
-- 	for i = 1, #rolesAssignments do
-- 		local ra = rolesAssignments[i]
-- 		for j = 1, #statesAssignments do
-- 			local sa = statesAssignments[j]
			
-- 			local a = {}
-- 			for r = 1, #ra do
-- 				table.insert(a, ra[r])
-- 			end
-- 			for s = 1, #sa do
-- 				table.insert(a, sa[s])
-- 			end

-- 			table.insert(assignments, a)
-- 		end
-- 	end

-- 	self.statesMemo[frameIndx] = assignments

-- 	return assignments
-- end

-- function getRoleToDetectionAssignments(numRoles, numDets)
-- 	local assignments = {}
-- 	if numRoles <= 1 then
-- 		for d = 1,numDets do
-- 			table.insert(assignments, {[1]=d})
-- 		end
-- 		return assignments
-- 	end

-- 	local subAssignments = getRoleToDetectionAssignments(numRoles-1, numDets)
-- 	for i = 1, #subAssignments do
-- 		local subAssmnt = subAssignments[i]
-- 		for d = 1, numDets do
-- 			local a = {}
-- 			for j = 1, #subAssmnt do
-- 				a[j] = subAssmnt[j]
-- 			end
-- 			table.insert(a, d)
-- 			table.insert(assignments, a)
-- 		end
-- 	end
-- 	return assignments
-- end

-- function SentenceTracker:getWordToStateAssignments(numWords)
-- 	local assignments = {}
-- 	if numWords == 1 then
-- 		for s = 1, self.words[1].stateTransitions:size(1) do
-- 			table.insert(assignments, {[1]=s})
-- 		end
-- 	else
-- 		local subAssignments = self:getWordToStateAssignments(numWords-1)
-- 		for i = 1, #subAssignments do
-- 			local subAssmnt = subAssignments[i]
-- 			for s = 1, self.words[numWords].stateTransitions:size(1) do
-- 				local a = {}
-- 				for j = 1, #subAssmnt do
-- 					a[j] = subAssmnt[j]
-- 				end
-- 				table.insert(a, s)
-- 				table.insert(assignments, a)
-- 			end
-- 		end
-- 	end
-- 	return assignments
-- end








