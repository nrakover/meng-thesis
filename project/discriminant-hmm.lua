require 'torch'
require 'svm'

DiscriminantHMM = {}

function DiscriminantHMM:new()
	newObj = {}
	self.__index = self
	return setmetatable(newObj, self)
end

function DiscriminantHMM:train(sequences, nStates, initClassifier, initTransitions, initPriors, maxIters, epsilon)
	-- Default parameters
	maxIters = maxIters or 8
	epsilon = epsilon or 1.0e-6

	self.nStates = nStates

	-- Init transitions
	self.trans = initTransitions or randomTransitionsMatrix(nStates)

	-- Init priors
	self.priors = initPriors or randomPriorsVector(nStates)

	-- Init classifier
	self.classifier = initClassifier

	-- Evaluate loglikelihood
	-- TODO

	print('Starting training')

	local iteration = 1
	while iteration <= maxIters do
		print('\n\n==> Iteration ', iteration)

		-- Estimate states
		local transitionCounts, priorCounts, stateSequences = self:E(sequences)

		-- Re-estimate parameters
		self:M(sequences, transitionCounts, priorCounts, stateSequences)

		-- Evaluate loglikelihood and break if appropriate
		-- TODO

		iteration = iteration + 1
	end
	
	return {nStates=self.nStates, transitionMatrix=self.trans, priors=self.priors, classifier=self.classifier}
end

local function DiscriminantHMM:E(sequences)
	-- Cumulative expected counts
	local transitionCounts = torch.zeros(self.nStates, self.nStates)
	local priorCounts = torch.zeros(self.nStates)

	-- Best state estimates for sequences
	local stateSequences = {}

	-- Iterate over sequences
	for i = 1, #sequences do
		local sequence = sequences[i]

		-- Estimate highest scoring state sequence
		-- NOTE: this calls DiscriminantHMM:setPseudoEmissions()
		stateSequences[i] = self:bestStateSequence(sequence)

		-- Sequence expected counts
		local sequenceTransCounts = torch.zeros(self.nStates, self.nStates)
		local sequencePriorCounts = torch.zeros(self.nStates)

		-- Set memo tables
		self:setAlphaBetaMemoTables(#sequence)

		-- Iterate over timeslices
		for t = 1, #sequence-1 do
			-- Iterate over starting states
			for p = 1, self.nStates do
				-- Iterate over ending states
				for q = 1, self.nStates do
					local posterior_p_to_q = self:alpha(p, t) * self.trans[p][q] * self:pseudoEmissionProb(p, t) * self:beta(q, t+1, #sequence)

					sequenceTransCounts[p][q] = sequenceTransCounts[p][q] + posterior_p_to_q

					-- When the state appears in the first position, contribute to the prior distribution
					if t == 1 then
						sequencePriorCounts[p] = sequencePriorCounts[p] + posterior_p_to_q
					end
				end
			end
		end

		-- Add sequence counts to global expected counts
		transitionCounts:add(sequenceTransCounts)
		priorCounts:add(sequencePriorCounts)
	end

	return transitionCounts, priorCounts, stateSequences
end

local function DiscriminantHMM:M(sequences, transitionCounts, priorCounts, stateSequences)
	-- Estimate transition probabilities
	local totalMass = torch.sum(transitionCounts)
	if totalMass == 0 then totalMass = 1 end -- Degenerate case
	transitionCounts:add(totalMass * 0.0001 / self.nStates^2) -- Smoothing factor to prevent 0 probability
	local sumsByRow = torch.sum(transitionCounts, 2)
	local normalizer = torch.diag(torch.ones(self.nStates):cdiv(sumsByRow))
	self.trans = torch.mm(normalizer, transitionCounts)

	-- Estimate prior probabilities
	totalMass = torch.sum(priorCounts)
	if totalMass == 0 then totalMass = 1 end -- Degenerate case
	priorCounts:add(totalMass * 0.0001 / self.nStates) -- Smoothing factor to prevent 0 probability
	totalMass = torch.sum(priorCounts)
	self.priors = priorCounts:div(totalMass)

	-- Train classifier based on best state estimates
	local trainingSet = prepareClassifierTrainingSet(sequences, stateSequences)
	self.classifier = liblinear.train(trainingSet, '-s 0 -q')
end

local function DiscriminantHMM:pseudoEmissionProb(p, j)
	local stateIndex = self.stateIndices[p]

	-- print('p(y|x) = ', self.emissions[j][stateIndex])
	-- print('p(y) = ', self.priors[p])

	return self.emissions[j][stateIndex] / self.priors[p]
end

local function DiscriminantHMM:alpha(p, j)
	-- Base case:
	if j == 1 then return self.priors[p] end

	-- Use memo table when possible
	if self.alphaMemoTable[p][j] ~= -1 then
		return self.alphaMemoTable[p][j]
	end

	local total = 0
	for q = 1, self.nStates do
		total = total + self:alpha(q, j-1) * self.trans[q][p] * self:pseudoEmissionProb(q, j-1)
	end

	-- Memoize
	self.alphaMemoTable[p][j] = total

	return total
end

local function DiscriminantHMM:beta(p, j, n)
	if j == n then return self:pseudoEmissionProb(p, j) end

	-- Use memo table when possible
	if self.betaMemoTable[p][j] ~= -1 then
		return self.betaMemoTable[p][j]
	end

	local total = 0
	for q = 1, self.nStates do
		total = total + self.trans[p][q] * self:beta(q, j+1, n)
	end
	total = total * self:pseudoEmissionProb(p, j)

	-- Memoize
	self.betaMemoTable[p][j] = total

	return total
end

function DiscriminantHMM:bestStateSequence(sequence)
	-- Set pseudo emisions
	self:setPseudoEmissions(sequence)

	-- Initialize memo table
	self.viterbiMemoTable = -torch.ones(self.nStates, #sequence)

	-- Run Viterbi algorithm
	local bestSequence = {}
	local bestPrevious = nil
	local bestScore = nil
	for v = 1, self.nStates do
		local score = self:viterbi(v, #sequence)
		if bestScore == nil or score > bestScore then
			bestScore = score
			bestPrevious = v
		end
	end
	bestSequence[#sequence] = bestPrevious

	-- Back-trace optimal states
	for t = #sequence-1, 1, -1 do
		bestScore = nil
		for v = 1, self.nStates do
			local score = self:viterbi(v, t) * self.trans[v][bestSequence[t+1]]
			if bestScore == nil or score > bestScore then
				bestScore = score
				bestPrevious = v
			end
		end
		bestSequence[t] = bestPrevious
	end

	return bestSequence
end

local function DiscriminantHMM:viterbi(v, k)
	-- Base case
	if k == 1 then return self.priors[v] * self:pseudoEmissionProb(v, k) end

	-- Use memo table when possible
	if self.viterbiMemoTable[v][k] ~= -1 then
		return self.viterbiMemoTable[v][k]
	end

	-- Compute optimal score
	local bestScore = nil
	for u = 1, self.nStates do
		local score = self:viterbi(u, k-1) * self.trans[u][v]
		if bestScore == nil or score > bestScore then bestScore = score end
	end
	bestScore = bestScore * self:pseudoEmissionProb(v, k)

	-- Memoize
	self.viterbiMemoTable[v][k] = bestScore

	return bestScore
end

local function DiscriminantHMM:setPseudoEmissions(sequence)
	local labels, accuracy, prob = liblinear.predict(sequence, self.classifier, '-b 1 -q')

	self.emissions = prob
	self.stateIndices = {}
	for i = 1, self.classifier.label:size(1) do
		local state = self.classifier.label[i]
		self.stateIndices[state] = i
	end
end

local function DiscriminantHMM:setAlphaBetaMemoTables(sequenceLength)
	self.alphaMemoTable = -torch.ones(self.nStates, sequenceLength)
	self.betaMemoTable = -torch.ones(self.nStates, sequenceLength)
end

local function randomTransitionsMatrix(nStates)
	local trans = torch.rand(nStates, nStates)
	trans:add(0.0001) -- Smoothing factor to prevent 0 probability
	local sumsByRow = torch.sum(trans, 2)
	local normalizer = torch.diag(torch.ones(nStates):cdiv(sumsByRow))
	return torch.mm(normalizer, trans)
end

local function randomPriorsVector(nStates)
	local priors = torch.rand(nStates)
	priors:add(0.0001) -- Smoothing factor to prevent 0 probability
	local total = torch.sum(priors)
	return priors:div(total)
end

local function prepareClassifierTrainingSet(sequences, stateSequences)
	-- Data set in SVMLight format
	local d = {}

	-- Datapoint index
	local i = 1

	for s = 1, #sequences do
		local observationSequence = sequences[s]
		local stateSequence = stateSequences[s]

		for t = 1, #observationSequence do
			local record = {}
			record[1] = stateSequence[t]
			record[2] = observationSequence[t][2]

			d[i] = record

			-- Increment the datapoint index
			i = i + 1
		end
	end

	return d
end
