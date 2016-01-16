require 'torch'

local matio = require 'matio'
matio.use_lua_strings = true

dofile('/local/nrakover/meng/project/tracker-hmm.lua')
dofile('/local/nrakover/meng/project/word-hmm.lua')


SentenceTracker = {}

function SentenceTracker:new(sentence, video_detections_path, video_features_path, video_optflow_path, word_models, filter_detections, words_to_filter_by)
	local newObj = {}
	self.__index = self
	newObj = setmetatable(newObj, self)

	-- Parse sentence
	newObj:parseSentence(sentence)

	-- Process video
	filter_detections = filter_detections or false
	newObj:processVideo(video_detections_path, video_features_path, video_optflow_path, filter_detections, word_models, words_to_filter_by)

	-- Initialize tracker
	newObj.tracker = Tracker:new(newObj.detectionsByFrame, newObj.detectionsOptFlow)

	-- Initialize word trackers
	newObj:buildWordModels(word_models)

	return newObj
end

function SentenceTracker:processVideo(video_detections_path, video_features_path, video_optflow_path, filter_detections, word_models, words_to_filter_by)
	-- Load the detection proposals
	self.detectionsByFrame = self:detectionsTensorToTable( matio.load(video_detections_path , 'detections_by_frame').detections )
	self.numFrames = #self.detectionsByFrame

	-- Load the neural network features from proposals
	self.detectionFeatures = torch.load(video_features_path)

	if filter_detections then
		self.detectionsByFrame, self.detectionFeatures = self:filterDetections( self.detectionsByFrame, self.detectionFeatures, word_models, words_to_filter_by, 4)
	end

	-- Load the optical flow for each frame
	self.detectionsOptFlow = torch.load(video_optflow_path)

end

function SentenceTracker:filterDetections( all_detections_by_frame, all_features, word_models, words_to_filter_by, K )
	local filtered_detections = {}
	local filtered_features = {}

	-- For each frame, select the best detections
	for fIndx = 1, #all_detections_by_frame do

		local scores_by_role = torch.Tensor(self.numRoles, #all_detections_by_frame[fIndx])
		-- Iterate over detections
		for detIndx = 1, #all_detections_by_frame[fIndx] do
			local features = torch.squeeze(all_features[fIndx][detIndx]:clone()):double()

			-- Score the detection
			for i,w in ipairs(self.positionToWord) do
				-- Only filter by 1-state words that take a single argument
				if #self.positionToRoles[i] == 1 and words_to_filter_by[w] ~= nil and word_models[w] ~= nil and word_models[w].priors:size(1) == 1 then
					local ll = math.log(word_models[w].emissions[1]:forward(features)[1])
					scores_by_role[self.positionToRoles[i][1]][detIndx] = scores_by_role[self.positionToRoles[i][1]][detIndx] + ll
				end
			end
		end

		filtered_detections[fIndx] = {}
		filtered_features[fIndx] = {}
		local detections_included = {}
		-- Select the best set per role
		for r = 1, scores_by_role:size(1) do
			-- Sort in descending order along the 2nd dimension
			local _, sorted_indices = torch.sort(scores_by_role[{{r},{}}], 2, true)

			-- Take the top K detections
			for i = 1, K do
				local detIndx = sorted_indices[1][i]
				if detections_included[detIndx] == nil then -- prevents repeated detections
					detections_included[detIndx] = true
					table.insert(filtered_detections[fIndx], all_detections_by_frame[fIndx][detIndx])
					table.insert(filtered_features[fIndx], all_features[fIndx][detIndx])
				end
			end
		end
	end

	return filtered_detections, filtered_features
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
		self.words[word] = Word:new(model.emissions, model.transitions, model.priors, self.detectionsByFrame, self.detectionFeatures, self.detectionsOptFlow)
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

	local v = self:startNode()
	local bestScore = nil
	local bestPath = nil
	while v ~= nil do
		local piResult = self:PI(self.numFrames, v)
		if bestScore == nil or piResult.score > bestScore then
			bestScore = piResult.score
			bestPath = piResult.path
		end
		v = self:nextNode(self.numFrames, v)
	end

	print('==> FINISHED')
	return bestPath, bestScore
end

function SentenceTracker:partialEStep( words_to_learn )
	-- Summary statistics to accumulate
	local state_transitions_by_word, priors_per_word, observations_per_word = self:initSummaryStatistics(words_to_learn)

	-- Initialize memo tables
	self.alphaMemo = {}
	self.betaMemo = {}

	-- Compute total log probability of the sequence
	local Z = self:logTotalProbabilityOfSequence()

	-- Iterate over frames
	for frameIndx = 1, self.numFrames - 1 do
		-- Iterate over the first node
		local p = self:startNode()
		while p ~= nil do
			-- Iterate over the second node
			local q = self:startNode()
			while q ~= nil do
				-- Compute adjacent node posteriors
				local transitions_ll = self:computeTracksTransitionScore( frameIndx+1, p, q ) + self:computeWordsTransitionScore( frameIndx+1, p, q )
				local observations_ll = self:computeTracksObservationScore( frameIndx, p ) + self:computeWordsObservationScore( frameIndx, p )
				local posterior_p_to_q = math.exp( ( self:logAlpha(frameIndx, p) + transitions_ll + observations_ll + self:logBeta(frameIndx+1, q) ) - Z )

				-- Accumulate the posterior
				state_transitions_by_word, priors_per_word, observations_per_word = self:accumulatePosterior( posterior_p_to_q, frameIndx, p, q, state_transitions_by_word, priors_per_word, observations_per_word )

				-- Next node
				q = self:nextNode(frameIndx+1, q)
			end

			-- Next node
			p = self:nextNode(frameIndx, p)
		end

		print('Done with frame '..frameIndx)
	end

	return state_transitions_by_word, priors_per_word, observations_per_word, Z
end

function SentenceTracker:accumulatePosterior( posterior, frameIndx, p, q, state_transitions_by_word, priors_per_word, observations_per_word )
	-- Iterate over sentence, accumulate only for words we want to learn
	for i,w in ipairs(self.positionToWord) do
		if state_transitions_by_word[w] ~= nil then
			local first_state = p[self.numRoles + i]
			local second_state = q[self.numRoles + i]

			-- Accumulate state transitions
			state_transitions_by_word[w][first_state][second_state] = state_transitions_by_word[w][first_state][second_state] + posterior


			-- Accumulate observations
			local detections = {}
			for j,r in ipairs(self.positionToRoles[i]) do
				detections[j] = p[r]
			end

			local obs_key = self.words[w]:getKey(first_state, frameIndx, detections)
			if observations_per_word[w][first_state][obs_key] == nil then
				local observation_features = self.words[w]:extractFeatures( frameIndx, detections )
				observations_per_word[w][first_state][obs_key] = {example=observation_features, weight=posterior}
			else
				observations_per_word[w][first_state][obs_key].weight = observations_per_word[w][first_state][obs_key].weight + posterior
			end


			-- If it's the first frame, accumulate state priors
			if frameIndx == 1 then
				priors_per_word[w][first_state] = priors_per_word[w][first_state] + posterior
			end


			-- If it's the last frame, accumulate observations for second state
			if frameIndx == self.numFrames - 1 then
				local detections = {}
				for j,r in ipairs(self.positionToRoles[i]) do
					detections[j] = q[r]
				end

				local obs_key = self.words[w]:getKey(second_state, frameIndx+1, detections)
				if observations_per_word[w][second_state][obs_key] == nil then
					local observation_features = self.words[w]:extractFeatures( frameIndx+1, detections )
					observations_per_word[w][second_state][obs_key] = {example=observation_features, weight=posterior}
				else
					observations_per_word[w][second_state][obs_key].weight = observations_per_word[w][second_state][obs_key].weight + posterior
				end
			end

		end
	end

	return state_transitions_by_word, priors_per_word, observations_per_word
end

function SentenceTracker:initSummaryStatistics( words_to_learn )
	local state_transitions_by_word = {}
	local priors_per_word = {}
	local observations_per_word = {}

	for i = 1, #words_to_learn do
		local w = words_to_learn[i]
		state_transitions_by_word[w] = torch.zeros(self.words[w].stateTransitions:size())
		priors_per_word[w] = torch.zeros(self.words[w].statePriors:size())
		observations_per_word[w] = {}
		for state = 1, self.words[w].statePriors:size(1) do
			observations_per_word[w][state] = {}
		end
	end

	return state_transitions_by_word, priors_per_word, observations_per_word
end

function SentenceTracker:logTotalProbabilityOfSequence()
	local Z = 0
	local p = self:startNode()
	while p ~= nil do
		Z = Z + math.exp( self:logAlpha(1,p) + self:logBeta(1,p) )
		-- Next node
		p = self:nextNode(1,p)
	end
	return math.log(Z)
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
		u = self:nextNode(k-1, u)
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
	if k == self.numFrames then
		local loglikelihood = self:computeTracksObservationScore( k, v ) + self:computeWordsObservationScore( k, v )
		-- Memoize and return
		self.betaMemo[key] = loglikelihood
		return loglikelihood
	end

	-- Recursive Case:
	local marginal_likelihood = 0
	-- Iterate over possibe next nodes
	local u = self:startNode()
	while u ~= nil do
		local transitions_ll = self:computeTracksTransitionScore( k+1, v, u ) + self:computeWordsTransitionScore( k+1, v, u )
		local observations_ll = self:computeTracksObservationScore( k, v ) + self:computeWordsObservationScore( k, v )
		local ll_for_u = self:logBeta( k+1, u) + transitions_ll + observations_ll

		-- Accumulate
		marginal_likelihood = marginal_likelihood + math.exp(ll_for_u)
		-- Next node
		u = self:nextNode(k+1, u)
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
			u = self:nextNode(k-1, u)
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

function SentenceTracker:nextNode( frameIndx, node )
	local nextNode = {}

	local carry = 1
	for i = #node, 1, -1 do
		if node[i] == self:maxValueAt(frameIndx, i) and carry == 1 then
			if i == 1 then return nil end
			if self:maxValueAt(frameIndx, i) == 0 then
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

function SentenceTracker:maxValueAt( frameIndx, i )
	if i <= self.numRoles then
		return #self.detectionsByFrame[frameIndx]
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

function SentenceTracker:detectionsTensorToTable( detections_by_frame )
	local detections_table = {}
	for fIndx = 1, detections_by_frame:size(1) do
		detections_table[fIndx] = {}
		for detIndx = 1, detections_by_frame:size(2) do
			detections_table[fIndx][detIndx] = detections_by_frame[fIndx][detIndx]:clone()
		end
	end
	return detections_table
end



