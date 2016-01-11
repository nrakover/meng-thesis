require 'torch'

dofile('/local/nrakover/meng/project/sentence-hmm.lua')
dofile('/local/nrakover/meng/classifiers.lua')

WordLearner = {}

function WordLearner:learnWords( words_to_learn, videos, sentences, labels, initial_word_models, max_iterations )
	local current_word_models = initial_word_models

	local maxIters = max_iterations or 10

	-- Perform iterations of EM
	for iter = 1, maxIters do

		-- E-Step: calculate posteriors
		local state_transitions_by_word, priors_per_word, observations_per_word = self:EStep( words_to_learn, videos, sentences, labels, current_word_models )

		-- M-Step: re-estimate word models
		current_word_models = self:MStep( current_word_models, words_to_learn, videos, labels, state_transitions_by_word, priors_per_word, observations_per_word )
	end

	return learned_word_models
end


function WordLearner:EStep( words_to_learn, videos, sentences, labels, current_word_models )
	local state_transitions_by_word, priors_per_word, observations_per_word = self:initSummaryStatistics( words_to_learn, current_word_models )

	-- Iterate over examples
	for i = 1, #videos do
		-- Instantiate sentence tracker
		local sentence_tracker = -- TODO: implement	$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

		-- Get posteriors for the example
		local trans, priors, obs = sentence_tracker:partialEStep( words_to_learn )

		-- Aggregate
		for j = 1, #words_to_learn do
			local w = words_to_learn[j]
			if labels[i] == 1 then
				state_transitions_by_word[w] = state_transitions_by_word[w] + trans[w]
				priors_per_word[w] = priors_per_word[w] + priors[w]
			else
				-- subtract counts for negative examples
				state_transitions_by_word[w] = state_transitions_by_word[w] - trans[w]
				priors_per_word[w] = priors_per_word[w] - priors[w]
			end

			-- Compile the observation examples for each word state
			for state = 1, priors:size(1) do
				for k,v in pairs(obs[w][state]) do
					table.insert(observations_per_word[w][state].examples, v.example)
					table.insert(observations_per_word[w][state].labels, labels[i])
					table.insert(observations_per_word[w][state].weights, v.weight)
				end
			end
		end
	end

	return state_transitions_by_word, priors_per_word, observations_per_word
end

function WordLearner:MStep( current_word_models, words_to_learn, videos, labels, state_transitions_by_word, priors_per_word, observations_per_word )
	local new_word_models = {}
	
	-- Iterate over words to learn
	for w,_ in ipairs(current_word_models) do

		-- Only compute models for words we mean to learn
		if words_to_learn[w] ~= nil then
			-- Compute the transition probabilities
			local state_transition_counts = state_transitions_by_word[w]
			local new_state_transitions = torch.FloatTensor(state_transition_counts:size())
			for p = 1, new_state_transitions:size(1) do
				local total_mass = state_transition_counts[{{p},{}}]:sum()
				new_state_transitions[{{p},{}}] = state_transition_counts[{{p},{}}] / total_mass
			end

			-- Compute the state priors
			local state_priors_counts = priors_per_word[w]
			local new_state_priors = state_priors_counts / state_priors_counts:sum()

			-- Compute the emissions models
			local new_emissions_models = {}
			for state = 1, state_priors_counts:size(1) do
				new_emissions_models[state] = trainLinearModel(observations_per_word[w][state].examples, torch.Tensor(observations_per_word[w][state].labels), observations_per_word[w][state].weights, true)
			end


			-- Update the model parameters
			new_word_models[w] = {}
			new_word_models[w].transitions = new_state_transitions
			new_word_models[w].priors = new_state_priors
			new_word_models[w].emissions = new_emissions_models
		else
			-- If we don't want to learn this word, reuse old model
			new_word_models[w] = current_word_models[w]
		end
	end

	return new_word_models
end

function WordLearner:initSummaryStatistics( words_to_learn, current_word_models )
	local state_transitions_by_word = {}
	local priors_per_word = {}
	local observations_per_word = {}

	for i = 1, #words_to_learn do
		local w = words_to_learn[i]
		state_transitions_by_word[w] = torch.zeros(current_word_models[w].transitions:size())
		priors_per_word[w] = torch.zeros(current_word_models[w].priors:size())
		observations_per_word[w] = {}
		for state = 1, current_word_models[w].priors:size(1) do
			observations_per_word[w][state] = {examples={}, labels={}, weights={}}
		end
	end

	return state_transitions_by_word, priors_per_word, observations_per_word
end
