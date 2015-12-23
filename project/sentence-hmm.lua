require 'torch'

local matio = require 'matio'
matio.use_lua_strings = true

dofile('/local/nrakover/meng/project/tracker-hmm.lua')
dofile('/local/nrakover/meng/project/word-hmm.lua')


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

function SentenceTracker:processVideo(video_detections_path, video_features_path, video_optflow_path)
	-- Load the detection proposals
	self.detectionsByFrame = matio.load(video_detections_path , 'detections_by_frame').detections

	-- Load the optical flow for each frame
	self.detectionsOptFlow = torch.load(video_optflow_path)

	-- Load the neural network features from proposals
	self.detectionFeatures = torch.load(video_features_path)
end

function SentenceTracker:parseSentence(sentence)
	self.positionToWord = {}
	self.positionToRoles = {}
	self.numRoles = 0
	for i = 1, #sentence do
		word_role_pair = sentence[i]
		self.positionToWord[i] = word_role_pair.word
		self.positionToRoles[i] = word_role_pair.roles
		for j = 1, #word_role_pair.roles do
			r = word_role_pair.roles[j]
			if r > self.numRoles then
				self.numRoles = r
			end
		end
	end
end

function SentenceTracker:buildWordModels(word_models)
	self.words = {}
	for word,model in pairs(word_models) do
		self.words[word] = Word:new(model.emissions, model.transitions, model.priors, self.detectionsByFrame, self.detectionFeatures)
	end
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

function SentenceTracker:partialEStep( words_to_learn )
	-- Initialize memo tables
	self.alphaMemo = {}
	self.betaMemo = {}

end

function SentenceTracker:logAlpha( k, v )
	-- Use this to index into memo table
	local key = self:getKey(k,v)

	-- Check if value is memoized
	if self.alphaMemo[key] ~= nil then
		return self.alphaMemo[key]
	end

	-- Base Case:
	if k == 1 then
		-- Word prior scores
		local loglikelihood = self:computeWordsPriorScore( v )
		-- Memoize and return
		self.alphaMemo[key] = loglikelihood
		return loglikelihood
	end

	-- Recursive Case:
	local marginal_likelihood = 0
	-- Iterate over possibe previous nodes
	local u = self:startNode()
	while u ~= nil do
		local transitions_ll = self:computeTracksTransitionScore( k, u, v ) + self:computeWordsTransitionScore( k, u, v )
		local observations_ll = self:computeTracksObservationScore( k-1, u ) + self:computeWordsObservationScore( k-1, u )
		local ll_for_u = self:logAlpha( k-1, u) + transitions_ll + observations_ll

		-- Accumulate
		marginal_likelihood = marginal_likelihood + math.exp(ll_for_u)
		-- Next node
		u = self:nextNode(u)
	end

	-- Memoize and return
	self.alphaMemo[key] = math.log(marginal_likelihood)
	return math.log(marginal_likelihood)

end

function SentenceTracker:logBeta( k, v )
	-- Use this to index into memo table
	local key = self:getKey(k,v)

	-- Check if value is memoized
	if self.betaMemo[key] ~= nil then
		return self.betaMemo[key]
	end

	-- Base Case:
	if k == self.detectionsByFrame:size(1) then
		local loglikelihood = self:computeTracksObservationScore( k, v ) + self:computeWordsObservationScore( k, v )
		-- Memoize and return
		self.betaMemo[key] = loglikelihood
		return loglikelihood
	end

	-- Recursive Case:
	local marginal_likelihood = 0
	-- Iterate over possibe previous nodes
	local u = self:startNode()
	while u ~= nil do
		local transitions_ll = self:computeTracksTransitionScore( k+1, v, u ) + self:computeWordsTransitionScore( k+1, v, u )
		local observations_ll = self:computeTracksObservationScore( k, v ) + self:computeWordsObservationScore( k, v )
		local ll_for_u = self:logBeta( k+1, u) + transitions_ll + observations_ll

		-- Accumulate
		marginal_likelihood = marginal_likelihood + math.exp(ll_for_u)
		-- Next node
		u = self:nextNode(u)
	end

	-- Memoize and return
	self.betaMemo[key] = math.log(marginal_likelihood)
	return math.log(marginal_likelihood)
end

function SentenceTracker:PI( k, v )
	-- Use this to index into memo table
	local key = self:getKey(k,v)

	-- Check if value is memoized
	if self.piMemo[key] ~= nil then
		return self.piMemo[key]
	end


	local scoreToReturn = nil
	local bestPath = nil

	-- ============================
	-- Calculate observation score
	-- 1. Tracker observation scores
	local tracksObservationScore = self:computeTracksObservationScore( k, v )

	-- 2. Words observation scores
	local wordsObservationScore = self:computeWordsObservationScore( k, v )


	-- Base Case:
	if k == 1 then
		-- Word prior scores
		local wordsPriorScore = self:computeWordsPriorScore( v )

		scoreToReturn = tracksObservationScore + wordsObservationScore + wordsPriorScore
		bestPath = {[1]=v}

	-- Recursive case:
	else
		-- Calculate best transition score

		-- Iterate over possibe previous nodes
		local u = self:startNode()
		local bestTransitionScore = nil
		local bestPathPrefix = nil
		while u ~= nil do			
			local prevResult = self:PI(k-1, u)
			
			-- 1. Tracker temporal coherence scores
			local tracksTransitionScore = self:computeTracksTransitionScore( k, u, v )

			-- 2. Word state transition scores 
			local wordsTransitionScore = self:computeWordsTransitionScore( k, u, v )

			-- Check if current score is best score
			if bestTransitionScore == nil or prevResult.score+tracksTransitionScore+wordsTransitionScore > bestTransitionScore then
				bestTransitionScore = prevResult.score+tracksTransitionScore+wordsTransitionScore
				bestPathPrefix = prevResult.path
			end

			-- Next node
			u = self:nextNode(u)
		end

		scoreToReturn = tracksObservationScore + wordsObservationScore + bestTransitionScore
		bestPath = {}
		for p = 1, #bestPathPrefix do
			table.insert(bestPath, bestPathPrefix[p])
		end
		table.insert(bestPath, v)
	end

	-- ============================

	local result = {score=scoreToReturn, path=bestPath}

	-- Memoize
	self.piMemo[key] = result

	print('Computed: '..key)

	return result
end

function SentenceTracker:computeTracksObservationScore( k, v )
	local tracksObservationScore = 0
	for r = 1, self.numRoles do
		tracksObservationScore = tracksObservationScore + self.tracker:detectionStrength(k, v[r])
	end
	return tracksObservationScore
end

function SentenceTracker:computeWordsObservationScore( k, v )
	local wordsObservationScore = 0
	for i,w in ipairs(self.positionToWord) do
		if v[i + self.numRoles] ~= 0 then
			local state = v[i + self.numRoles]
			local detections = {}
			for j,r in ipairs(self.positionToRoles[i]) do
				detections[j] = v[r]
			end
			wordsObservationScore = wordsObservationScore + math.log(self.words[w]:probOfEmission(state, k, detections))
		end
	end
	return wordsObservationScore
end

function SentenceTracker:computeWordsPriorScore( v )
	local wordsPriorScore = 0
	for i,w in ipairs(self.positionToWord) do
		if v[i + self.numRoles] ~= 0 then
			local state = v[i + self.numRoles]
			wordsPriorScore = wordsPriorScore + math.log(self.words[w]:statePrior(state))
		end
	end
	return wordsPriorScore
end

function SentenceTracker:computeTracksTransitionScore( k, u, v )
	local tracksTransitionScore = 0
	for r = 1, self.numRoles do
		tracksTransitionScore = tracksTransitionScore + self.tracker:temporalCoherence(k, u[r], v[r])
	end
	return tracksTransitionScore
end

function SentenceTracker:computeWordsTransitionScore( k, u, v )
	local wordsTransitionScore = 0
	for i,w in ipairs(self.positionToWord) do
		if v[i + self.numRoles] ~= 0 then
			local state = v[i + self.numRoles]
			local prevState = u[i + self.numRoles]
			wordsTransitionScore = wordsTransitionScore + math.log(self.words[w]:probOfTransition(prevState, state))
		end
	end
	return wordsTransitionScore
end

function SentenceTracker:setPIMemoTable()
	self.piMemo = {}
end

function SentenceTracker:startNode()
	local node = {}
	for i = 1, self.numRoles do
		node[i] = 1
	end
	for i,w in ipairs(self.positionToWord) do
		if self.words[w] ~= nil then
			node[i + self.numRoles] = 1
		else
			node[i + self.numRoles] = 0
		end
	end
	return node
end

function SentenceTracker:nextNode( node )
	local nextNode = {}

	local carry = 1
	for i = #node, 1, -1 do
		if node[i] == self:maxValueAt(i) and carry == 1 then
			if i == 1 then return nil end
			if self:maxValueAt(i) == 0 then
				nextNode[i] = 0
			else
				nextNode[i] = 1
			end
		else
			nextNode[i] = node[i] + carry
			carry = 0
		end
	end
	return nextNode
end

function SentenceTracker:maxValueAt( i )
	if i <= self.numRoles then
		return self.detectionsByFrame:size(2)
	else
		local w = self.positionToWord[i - self.numRoles]
		if self.words[w] ~= nil then
			return self.words[w].stateTransitions:size(1)
		else
			return 0
		end
	end
end

function SentenceTracker:getKey(k, v)
	local key = (''..k)..':'
	for i = 1, #v do
		key = (key..v[i])..'_'
	end
	return key
end





