require 'torch'

local matio = require 'matio'
matio.use_lua_strings = true

dofile('/local/nrakover/meng/project/tracker-hmm.lua')
dofile('/local/nrakover/meng/project/word-hmm.lua')


SentenceTracker = {}


-- ##########################################
-- #####          INITIALIZATION         ####
-- ##########################################

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
	newObj.tracker = Tracker:new(newObj.detectionsByFrame, newObj.detectionsOptFlow, 80)

	-- Initialize word trackers
	newObj:buildWordModels(word_models)

	return newObj
end

function SentenceTracker:processVideo(video_detections_path, video_features_path, video_optflow_path, filter_detections, word_models, words_to_filter_by)
	local detections_struct = matio.load(video_detections_path , 'detections_by_frame')

	-- Load the detection proposals
	self.detectionsByFrame = detections_struct.detections
	
	-- Load the neural network features from proposals
	self.detectionFeatures = torch.load(video_features_path)

	-- Convert detections tensor into a table and filter out null detections (and corresponding features)
	self.detectionsByFrame, self.detectionFeatures, person_detector_indices = self:detectionsTensorToTable(self.detectionsByFrame, self.detectionFeatures, detections_struct.person_detector_indices)

	self.numFrames = #self.detectionsByFrame

	-- Load the optical flow for each frame
	self.detectionsOptFlow = torch.load(video_optflow_path)

	if filter_detections then
		self.detectionsByFrame, self.detectionFeatures, self.detectionIndicesPerRole = self:filterDetections( self.detectionsByFrame, self.detectionFeatures, self.detectionsOptFlow, word_models, words_to_filter_by, 4, person_detector_indices )
	else
		self.detectionIndicesPerRole = {}
		for f = 1, self.numFrames do
			self.detectionIndicesPerRole[f] = {}
			for r = 1, self.numRoles do
				self.detectionIndicesPerRole[f][r] = {}  --{r}
				for i = 1, #self.detectionsByFrame[f] do
					table.insert(self.detectionIndicesPerRole[f][r], i)
				end
			end
		end
	end
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


-- ##########################################
-- #####          MAP ESTIMATION         ####
-- ##########################################

function SentenceTracker:getBestTrack()
	local path, score = self:getBestPath()

	local track = {}
	for frameIndx = 1, #path do
		track[frameIndx] = {}
		local state = path[frameIndx]
		for r = 1, self.numRoles do
			local detIndx = self.detectionIndicesPerRole[frameIndx][r][state[r]]
			table.insert(track[frameIndx], self.detectionsByFrame[frameIndx][detIndx]:clone())
		end
	end
	return track, score
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


-- ##########################################
-- #####        PARTIAL BAUM-WELCH       ####
-- ##########################################

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
				local posterior_p_to_q = math.exp( (((-Z + self:logAlpha(frameIndx, p)) + transitions_ll) + observations_ll) + self:logBeta(frameIndx+1, q)  )

				-- Accumulate the posterior
				state_transitions_by_word, priors_per_word, observations_per_word = self:accumulatePosterior( posterior_p_to_q, frameIndx, p, q, state_transitions_by_word, priors_per_word, observations_per_word )

				-- Next node
				q = self:nextNode(frameIndx+1, q)
			end

			-- Next node
			p = self:nextNode(frameIndx, p)
		end
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
				detections[j] = self.detectionIndicesPerRole[frameIndx][r][p[r]]
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
					detections[j] = self.detectionIndicesPerRole[frameIndx+1][r][q[r]]
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
	-- local Z = 0
	local p = self:startNode()
	local list_for_LSE = {}
	while p ~= nil do
		table.insert(list_for_LSE, self:logAlpha(1,p) + self:logBeta(1,p))
		-- Z = Z + math.exp( self:logAlpha(1,p) + self:logBeta(1,p) )
		-- Next node
		p = self:nextNode(1,p)
	end
	-- print('LSE = '..self:logSumExp(list_for_LSE))
	-- print('Z = '..math.log(Z))
	-- return math.log(Z)
	return self:logSumExp(list_for_LSE)
end

function SentenceTracker:logSumExp( X_vals )
	local X = torch.DoubleTensor(X_vals)
	local max_x = X:max()

	if not (max_x > math.log(0)) then return math.log(0) end -- edge case when all elements are -inf

	return max_x + math.log( (X[torch.gt(X, math.log(0))] - max_x):exp():sum() )
end


-- ##########################################
-- #####        FORWARDS-BACKWARDS       ####
-- ##########################################

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
	local list_for_LSE = {}
	-- Iterate over possibe previous nodes
	local u = self:startNode()
	while u ~= nil do
		local transitions_ll = self:computeTracksTransitionScore( k, u, v ) + self:computeWordsTransitionScore( k, u, v )
		local observations_ll = self:computeTracksObservationScore( k-1, u ) + self:computeWordsObservationScore( k-1, u )
		local ll_for_u = self:logAlpha( k-1, u) + transitions_ll + observations_ll

		-- Accumulate
		table.insert(list_for_LSE, ll_for_u)
		-- Next node
		u = self:nextNode(k-1, u)
	end

	-- Memoize and return
	self.alphaMemo[key] = self:logSumExp(list_for_LSE)
	return self.alphaMemo[key]

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
		local loglikelihood = self:computeTracksObservationScore( k, v ) + self:computeWordsObservationScore( k, v ) + self:computeWordsTerminationScore( v )
		-- Memoize and return
		self.betaMemo[key] = loglikelihood
		return loglikelihood
	end

	-- Recursive Case:
	local list_for_LSE = {}
	-- Iterate over possibe next nodes
	local u = self:startNode()
	while u ~= nil do
		local transitions_ll = self:computeTracksTransitionScore( k+1, v, u ) + self:computeWordsTransitionScore( k+1, v, u )
		local observations_ll = self:computeTracksObservationScore( k, v ) + self:computeWordsObservationScore( k, v )
		local ll_for_u = self:logBeta( k+1, u) + transitions_ll + observations_ll

		-- if k == self.numFrames - 2 then print('transition: ', transitions_ll) end
		-- if k == self.numFrames - 2 then print('observations: ', observations_ll) end
		-- if k == self.numFrames - 2 then print('total: ', ll_for_u) end

		-- Accumulate
		table.insert(list_for_LSE, ll_for_u)
		-- Next node
		u = self:nextNode(k+1, u)
	end

	-- if k == self.numFrames - 2 then print(key, self:logSumExp(list_for_LSE)) end

	-- Memoize and return
	self.betaMemo[key] = self:logSumExp(list_for_LSE)
	return self.betaMemo[key]
end


-- ##########################################
-- #####             VITERBI             ####
-- ##########################################

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

		-- Account for terminal distribution on last frame
		if k == self.numFrames then
			scoreToReturn = scoreToReturn + self:computeWordsTerminationScore( v )
		end

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


-- ##########################################
-- #####      SCORING LATTICE NODES      ####
-- ##########################################

function SentenceTracker:computeTracksObservationScore( k, v )
	local tracksObservationScore = 0
	for r = 1, self.numRoles do
		local detIndx = self.detectionIndicesPerRole[k][r][v[r]]
		tracksObservationScore = tracksObservationScore + self.tracker:detectionStrength(k, detIndx)
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
				detections[j] = self.detectionIndicesPerRole[k][r][v[r]]
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

function SentenceTracker:computeWordsTerminationScore( v )
	local wordsTerminationScore = 0
	for i,w in ipairs(self.positionToWord) do
		if v[i + self.numRoles] ~= 0 then
			local state = v[i + self.numRoles]
			wordsTerminationScore = wordsTerminationScore + math.log(self.words[w]:stateTerminalDistribution(state))
		end
	end
	return wordsTerminationScore
end

function SentenceTracker:computeTracksTransitionScore( k, u, v )
	local tracksTransitionScore = 0
	for r = 1, self.numRoles do
		local detIndx = self.detectionIndicesPerRole[k][r][v[r]]
		local prevDetIndx = self.detectionIndicesPerRole[k-1][r][u[r]]
		tracksTransitionScore = tracksTransitionScore + self.tracker:temporalCoherence(k, prevDetIndx, detIndx)
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


-- ##########################################
-- #####           MEMOIZATION           ####
-- ##########################################

function SentenceTracker:setPIMemoTable()
	self.piMemo = {}
end

function SentenceTracker:getKey(k, v)
	local key = (''..k)..':'
	for i = 1, #v do
		key = (key..v[i])..'_'
	end
	return key
end


-- ##########################################
-- #####      LATTICE NODE ITERATION     ####
-- ##########################################

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
		return #self.detectionIndicesPerRole[frameIndx][i]
	else
		local w = self.positionToWord[i - self.numRoles]
		if self.words[w] ~= nil then
			return self.words[w].stateTransitions:size(1)
		else
			return 0
		end
	end
end


-- ##########################################
-- #####    FILTERING OBJECT PROPOSALS   ####
-- ##########################################

function SentenceTracker:filterDetections( all_detections_by_frame, all_features, optical_flow, word_models, words_to_filter_by, num_desired_proposals_per_role, person_detection_indices )
	
	-- Do first pass, overgenerating for each noun using forward-projection, without scoring two-argument words
	local filtered_detections, filtered_features, detection_indices_per_role = self:doSinglePassOfFilteringWithOptions( all_detections_by_frame, all_features, optical_flow, word_models, words_to_filter_by, 2*num_desired_proposals_per_role, true, false, 25, person_detection_indices )

	-- Do second pass, scoring with two-argument words, no forward-projection and narrowing down to the desired number of proposals
	-- filtered_detections, filtered_features, detection_indices_per_role = self:doSinglePassOfFilteringWithOptions( filtered_detections, filtered_features, optical_flow, word_models, words_to_filter_by, 5*num_desired_proposals_per_role, false, false, 20 )

	return filtered_detections, filtered_features, self:nonMaximalSuppression( detection_indices_per_role, filtered_detections, num_desired_proposals_per_role )
end

function SentenceTracker:doSinglePassOfFilteringWithOptions( all_detections_by_frame, all_features, optical_flow, word_models, words_to_filter_by, K, do_forward_project, score_with_two_arg_words, tracker_exponent, person_detection_indices )
	local filtered_detections = {}
	local filtered_features = {}
	local detection_indices_per_role = {}

	local scores_by_role_per_frame = {}
	local tracker = Tracker:new(all_detections_by_frame, optical_flow, tracker_exponent)

	local person_roles = {}
	if person_detection_indices ~= nil then
		for i,w in ipairs(self.positionToWord) do
			if w == 'person' then
				person_roles[self.positionToRoles[i][1]] = true
			end
		end
	end

	-- Score detections with word models
	for fIndx = 1, self.numFrames do
		
		-- Score with single-argument, single-state words and the tracker
		scores_by_role_per_frame[fIndx] = torch.Tensor(self.numRoles, #all_detections_by_frame[fIndx])

		-- Knock out non-person detections for all person roles
		for r = 1, self.numRoles do
			if person_roles[r] then
				scores_by_role_per_frame[fIndx][{{r}, {1, person_detection_indices[fIndx][1]-1}}] = math.log(0)
			end
		end

		-- Iterate over detections
		for detIndx = 1, #all_detections_by_frame[fIndx] do
			local features = torch.cat( torch.squeeze(all_features[fIndx][detIndx]:clone()):double(), torch.DoubleTensor({0,0,0}), 1 ) -- pad with optical flow features, which will be ignored

			-- Score the detection with the single-argument word models
			scores_by_role_per_frame = self:scoreWithStaticWords( fIndx, detIndx, features, scores_by_role_per_frame, word_models, words_to_filter_by )
		end

		-- Score with two-argument words
		if score_with_two_arg_words and fIndx == 1 then
			scores_by_role_per_frame = self:scoreWithTwoArgumentWords( scores_by_role_per_frame, all_detections_by_frame, all_features, optical_flow, word_models, words_to_filter_by )
		end
	end

	-- Score the detection with the tracker
	for fIndx = self.numFrames, 2, -1 do
		-- Iterate over detections
		for detIndx = 1, #all_detections_by_frame[fIndx] do
			scores_by_role_per_frame = self:scoreWithTracker( fIndx, detIndx, scores_by_role_per_frame, tracker, all_detections_by_frame )
		end
	end

	-- For each frame, select the best detections
	for fIndx = 1, self.numFrames do
		-- Take top K from each role
		filtered_detections, filtered_features, detection_indices_per_role = self:getTopKPerRole( K, fIndx, detection_indices_per_role, filtered_detections, filtered_features, scores_by_role_per_frame, all_detections_by_frame, all_features )
	end

	-- Add forward-projected proposals
	if do_forward_project then
		filtered_detections, filtered_features, detection_indices_per_role = self:addForwardProjectedProposals( tracker, detection_indices_per_role, filtered_detections, filtered_features, all_detections_by_frame, all_features, optical_flow )
	end

	return filtered_detections, filtered_features, detection_indices_per_role
end

function SentenceTracker:scoreWithStaticWords( fIndx, detIndx, features, scores_by_role_per_frame, word_models, words_to_filter_by )
	for i,w in ipairs(self.positionToWord) do
		-- Only filter by 1-state words that take a single argument
		if #self.positionToRoles[i] == 1 and words_to_filter_by[w] ~= nil and word_models[w] ~= nil and word_models[w].priors:size(1) == 1 then
			local ll = math.log(word_models[w].emissions[1]:forward(features)[1])
			scores_by_role_per_frame[fIndx][self.positionToRoles[i][1]][detIndx] = scores_by_role_per_frame[fIndx][self.positionToRoles[i][1]][detIndx] + ll
		end
	end
	return scores_by_role_per_frame
end

function SentenceTracker:scoreWithTracker( fIndx, detIndx, scores_by_role_per_frame, tracker, all_detections_by_frame )
	for r = 1, self.numRoles do
		local best_tracker_score = math.log(0)
		for prevDetIndx = 1, #all_detections_by_frame[fIndx-1] do
			local score = tracker:temporalCoherence(fIndx, prevDetIndx, detIndx) + scores_by_role_per_frame[fIndx-1][r][prevDetIndx]
			if score >= best_tracker_score then
				best_tracker_score = score
			end
		end
		scores_by_role_per_frame[fIndx][r][detIndx] = scores_by_role_per_frame[fIndx][r][detIndx] + best_tracker_score
	end
	return scores_by_role_per_frame
end

function SentenceTracker:scoreWithTwoArgumentWords( scores_by_role_per_frame, all_detections_by_frame, all_features, optical_flow, word_models, words_to_filter_by )
	local word_state = 1
	local fIndx = 1
	-- Iterate over two-argument words
	for i,w in ipairs(self.positionToWord) do
		if #self.positionToRoles[i] == 2 and words_to_filter_by[w] ~= nil and word_models[w] ~= nil then
			local model = word_models[w]
			local word = Word:new(model.emissions, model.transitions, model.priors, all_detections_by_frame, all_features, optical_flow)
			local wordScores = torch.Tensor(#all_detections_by_frame[fIndx], #all_detections_by_frame[fIndx])
			-- Iterate over 1st role
			for r1DetIndx = 1, #all_detections_by_frame[fIndx] do
				-- Iterate over 2nd role
				for r2DetIndx = 1, #all_detections_by_frame[fIndx] do
					wordScores[r1DetIndx][r2DetIndx] = math.log( word:probOfEmission(word_state, fIndx, {r1DetIndx, r2DetIndx}) )
				end
			end

			-- Score each detection
			local detection_scores_per_role = torch.Tensor(2, #all_detections_by_frame[fIndx])
			local r1 = self.positionToRoles[i][1]
			local r2 = self.positionToRoles[i][2]
			for r1DetIndx = 1, #all_detections_by_frame[fIndx] do
				detection_scores_per_role[1][r1DetIndx] = torch.add( wordScores[{{r1DetIndx},{}}], scores_by_role_per_frame[fIndx][{{r2},{}}] ):max()
			end
			for r2DetIndx = 1, #all_detections_by_frame[fIndx] do
				detection_scores_per_role[2][r2DetIndx] = torch.add( wordScores[{{},{r2DetIndx}}], scores_by_role_per_frame[fIndx][{{r1},{}}] ):max()
			end

			for r1DetIndx = 1, #all_detections_by_frame[fIndx] do
				scores_by_role_per_frame[fIndx][r1][r1DetIndx] = scores_by_role_per_frame[fIndx][r1][r1DetIndx] + detection_scores_per_role[1][r1DetIndx]
			end
			for r2DetIndx = 1, #all_detections_by_frame[fIndx] do
				scores_by_role_per_frame[fIndx][r2][r2DetIndx] = scores_by_role_per_frame[fIndx][r2][r2DetIndx] + detection_scores_per_role[2][r2DetIndx]
			end
		end
	end
	return scores_by_role_per_frame
end

function SentenceTracker:getTopKPerRole( K, fIndx, detection_indices_per_role, filtered_detections, filtered_features, scores_by_role_per_frame, all_detections_by_frame, all_features )
	filtered_detections[fIndx] = {}
	filtered_features[fIndx] = {}
	detection_indices_per_role[fIndx] = {}
	local detections_included = {}
	-- Select the best set per role
	for r = 1, self.numRoles do
		-- Sort in descending order along the 2nd dimension
		local _, sorted_indices = torch.sort(scores_by_role_per_frame[fIndx][{{r},{}}], 2, true)

		detection_indices_per_role[fIndx][r] = {}
		-- Take the top K detections
		for i = 1, math.min(K,sorted_indices:size(2)) do
			local detIndx = sorted_indices[1][i]
			if detections_included[detIndx] == nil then -- prevents repeated detections
				detections_included[detIndx] = #filtered_detections[fIndx] + 1
				table.insert(filtered_detections[fIndx], all_detections_by_frame[fIndx][detIndx])
				table.insert(filtered_features[fIndx], all_features[fIndx][detIndx])
			end
			table.insert(detection_indices_per_role[fIndx][r], detections_included[detIndx])
		end
	end
	return filtered_detections, filtered_features, detection_indices_per_role
end

function SentenceTracker:addForwardProjectedProposals( tracker, detection_indices_per_role, filtered_detections, filtered_features, all_detections_by_frame, all_features, optical_flow )
	for fIndx = self.numFrames, 2, -1 do
		local detections_included = {}
		local detections_included_per_role = {}
		for r = 1, self.numRoles do
			detections_included_per_role[r] = {}
			for i = 1, #detection_indices_per_role[fIndx][r] do
				local detIndx = detection_indices_per_role[fIndx][r][i]
				if detections_included_per_role[r][detIndx] == nil then
					detections_included_per_role[r][detIndx] = true
				end

				if detections_included[filtered_detections[fIndx][detIndx]] == nil then
					detections_included[filtered_detections[fIndx][detIndx]] = detIndx
				end
			end
		end
		local previous_detections_included_per_role = {}
		for r = 1, self.numRoles do
			previous_detections_included_per_role[r] = {}
			for i = 1, #detection_indices_per_role[fIndx][r] do
				local detIndx = detection_indices_per_role[fIndx-1][r][i]
				if previous_detections_included_per_role[r][detIndx] == nil then
					previous_detections_included_per_role[r][detIndx] = true
				end
			end
		end

		-- for i = 1, #filtered_detections[fIndx] do
		-- 	detections_included[filtered_detections[fIndx][i]] = true
		-- end
		for i = 1, #filtered_detections[fIndx-1] do
			-- Get bounds
			local old_bounds = filtered_detections[fIndx-1][i]:clone()
			if old_bounds[4] == old_bounds[2] then old_bounds[4] = old_bounds[4] + 1 end
			if old_bounds[3] == old_bounds[1] then old_bounds[3] = old_bounds[3] + 1 end
			old_bounds[3] = math.min( optical_flow[fIndx].flow_x:size(2), old_bounds[3] )
			old_bounds[4] = math.min( optical_flow[fIndx].flow_x:size(1), old_bounds[4] )

			-- Get optical flow
			local avg_flow_x = tracker:extractAvgFlowFromDistanceTransform(optical_flow[fIndx].flow_x, old_bounds[1], old_bounds[2], old_bounds[3], old_bounds[4])
			local avg_flow_y = tracker:extractAvgFlowFromDistanceTransform(optical_flow[fIndx].flow_y, old_bounds[1], old_bounds[2], old_bounds[3], old_bounds[4])
			local avg_flow = torch.Tensor( { avg_flow_x, avg_flow_y } )

			assert(avg_flow_x < -math.log(0), 'oops A')
			assert(avg_flow_y < -math.log(0), 'oops B')

			-- Project bounds
			local projected_proposal = self:forwardProject(old_bounds, avg_flow, optical_flow[fIndx].flow_x:size(2), optical_flow[fIndx].flow_x:size(1))

			-- Find best match
			local best_projected_index = self:findClosestProposal( projected_proposal, all_detections_by_frame[fIndx] )

			-- Add to the proposals for next frame
			if detections_included[all_detections_by_frame[fIndx][best_projected_index]] == nil and all_detections_by_frame[fIndx][best_projected_index] ~= nil then -- prevents repeated detections
				detections_included[all_detections_by_frame[fIndx][best_projected_index]] = #filtered_detections[fIndx] + 1
				table.insert(filtered_detections[fIndx], all_detections_by_frame[fIndx][best_projected_index])
				table.insert(filtered_features[fIndx], all_features[fIndx][best_projected_index])
			
			-- For debugging purposes
			elseif all_detections_by_frame[fIndx][best_projected_index] == nil then
				print('-----------------------------------------------------------------------')
				print('WARNING: unusual result in SentenceTracker:addForwardProjectedProposals')
				print('best_projected_index:')
				print(best_projected_index)
				print('fIndx:')
				print(fIndx)
				print('number of candidates:')
				print(#all_detections_by_frame[fIndx])
				print('-----------------------------------------------------------------------')
			end

			-- Add to all the appropriate roles
			local detIndx = detections_included[all_detections_by_frame[fIndx][best_projected_index]]
			for r = 1, self.numRoles do
				-- Check if the previous detection was included in the role
				if previous_detections_included_per_role[r][i] then
					-- Check if the new detection is already included
					if not detections_included_per_role[r][detIndx] then
						detections_included_per_role[r][detIndx] = true
						table.insert(detection_indices_per_role[fIndx][r], detIndx)
					end
				end
			end
		end
	end
	return filtered_detections, filtered_features, detection_indices_per_role
end

function SentenceTracker:forwardProject( old_proposal, flow_vector, max_width, max_height )
	local new_proposal = torch.DoubleTensor(4)
	new_proposal[1] = old_proposal[1] + flow_vector[1]
	new_proposal[2] = old_proposal[2] + flow_vector[2]
	new_proposal[3] = old_proposal[3] + flow_vector[1]
	new_proposal[4] = old_proposal[4] + flow_vector[2]

	new_proposal[{{1}}]:clamp(1,max_width)
	new_proposal[{{2}}]:clamp(1,max_height)
	new_proposal[{{3}}]:clamp(1,max_width)
	new_proposal[{{4}}]:clamp(1,max_height)

	return new_proposal
end

function SentenceTracker:findClosestProposal( projected_proposal, candidates )
	local shortest_distance = -math.log(0)
	local closest_proposal = nil
	for i = 1, #candidates do
		local candidate_proposal = candidates[i]
		local d = torch.dist(projected_proposal, candidate_proposal)
		if d < shortest_distance then
			shortest_distance = d
			closest_proposal = i
		end
	end
	return closest_proposal
end

function SentenceTracker:detectionsTensorToTable( detections_by_frame, detection_features, person_detector_indices )
	local detections_table = {}
	local filtered_detection_features = {}
	for fIndx = 1, detections_by_frame:size(1) do
		detections_table[fIndx] = {}
		filtered_detection_features[fIndx] = {}
		local num_null_detections = 0
		for detIndx = 1, detections_by_frame:size(2) do
			if not self:isNullDetection(detections_by_frame[fIndx][detIndx]) then
				table.insert(detections_table[fIndx], detections_by_frame[fIndx][detIndx]:clone())
				table.insert(filtered_detection_features[fIndx], detection_features[fIndx][detIndx]:clone())
			elseif person_detector_indices ~= null and detIndx < person_detector_indices[fIndx][1] then
				num_null_detections = num_null_detections + 1
			end
		end
		if person_detector_indices ~= null then
			person_detector_indices[fIndx][1] = person_detector_indices[fIndx][1] - num_null_detections
			person_detector_indices[fIndx][2] = person_detector_indices[fIndx][2] - num_null_detections
			assert(person_detector_indices[fIndx][2] == #detections_table[fIndx], ('person_detector_indices[fIndx][2] = '..person_detector_indices[fIndx][2])..(' , #detections_table[fIndx] = '..#detections_table[fIndx]))
		end
	end
	return detections_table, filtered_detection_features, person_detector_indices
end

function SentenceTracker:isNullDetection( detection )
	-- Get detection bounds
	local x_min = detection[1]
	local y_min = detection[2]
	local x_max = detection[3]
	local y_max = detection[4]

	return (x_min == x_max and y_min == y_max)
end

function SentenceTracker:nonMaximalSuppression( detection_indices_per_role, filtered_detections, num_desired_proposals_per_role, tracks_to_omit )
	-- Detection indices after non-maximal supression
	local suppressed_detection_indices_per_role = {}

	-- Iterate over each frame
	for fIndx = 1, self.numFrames do
		suppressed_detection_indices_per_role[fIndx] = {}
		-- Iterate over each role
		for r = 1, self.numRoles do
			suppressed_detection_indices_per_role[fIndx][r] = {}
			local knocked_out_detections = {} -- false/nil for values not knocked out, true for knocked out values
			local next_viable_indx = 1

			-- Optionally knock out the specified tracks
			if tracks_to_omit ~= nil then
				-- Iterate over forbidden detections
				for i = 1, #tracks_to_omit[fIndx] do
					local suppressing_box = tracks_to_omit[fIndx][i]
					-- Suppress the specified proposals
					for j = 1, #detection_indices_per_role[fIndx][r] do
						if not knocked_out_detections[j] then
							local box_to_suppress = filtered_detections[fIndx][detection_indices_per_role[fIndx][r][j]]
							knocked_out_detections[j] = self:shouldSuppress( suppressing_box, box_to_suppress, 0.01 )
						end
					end
				end
			end

			-- Select the indices
			for i = 1, num_desired_proposals_per_role do
				-- Find the best index that hasn't been knocked out
				for j = next_viable_indx, #detection_indices_per_role[fIndx][r] do
					if not knocked_out_detections[j] then
						-- Add it to the selected indices
						table.insert(suppressed_detection_indices_per_role[fIndx][r], detection_indices_per_role[fIndx][r][j])
						-- Supress the indices below it
						local suppressing_box = filtered_detections[fIndx][detection_indices_per_role[fIndx][r][j]]
						for k = j + 1, #detection_indices_per_role[fIndx][r] do
							if not knocked_out_detections[k] then
								local box_to_suppress = filtered_detections[fIndx][detection_indices_per_role[fIndx][r][k]]
								knocked_out_detections[k] = self:shouldSuppress( suppressing_box, box_to_suppress )
							end
						end

						next_viable_indx = j + 1
						break
					end
				end
			end
		end
	end
	return suppressed_detection_indices_per_role
end

function SentenceTracker:shouldSuppress( box_a, box_b, threshold )
	threshold = threshold or 0.5

	local A_x_min = box_a[1]
	local A_y_min = box_a[2]
	local A_x_max = box_a[3]
	local A_y_max = box_a[4]

	local B_x_min = box_b[1]
	local B_y_min = box_b[2]
	local B_x_max = box_b[3]
	local B_y_max = box_b[4]

	local I_x_min = math.max(A_x_min, B_x_min)
	local I_x_max = math.min(A_x_max, B_x_max)
	local I_y_min = math.max(A_y_min, B_y_min)
	local I_y_max = math.min(A_y_max, B_y_max)
	local intersection = math.max(0, I_x_max - I_x_min) * math.max(0, I_y_max - I_y_min)

	local union = (A_x_max - A_x_min) * (A_y_max - A_y_min)  +  (B_x_max - B_x_min) * (B_y_max - B_y_min)  -  intersection

	local IoU = intersection / union

	if union == 0 then return true end -- edge case

	return IoU > threshold
end


