require 'torch'

dofile('/local/nrakover/meng/project/sentence-hmm.lua')

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

				observations_per_word[w][i] = obs[w]
			else
				-- subtract counts for negative examples
				state_transitions_by_word[w] = state_transitions_by_word[w] - trans[w]
				priors_per_word[w] = priors_per_word[w] - priors[w]

				observations_per_word[w][i] = obs[w]
			end
		end
	end

	return state_transitions_by_word, priors_per_word, observations_per_word
end

function WordLearner:MStep( current_word_models, words_to_learn, videos, labels, state_transitions_by_word, priors_per_word, observations_per_word )
	-- TODO: implement	$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
end

function WordLearner:initSummaryStatistics( words_to_learn, current_word_models )
	local state_transitions_by_word = {}
	local priors_per_word = {}
	local observations_per_word = {}

	for i = 1, #words_to_learn do
		local w = words_to_learn[i]
		state_transitions_by_word[w] = torch.zeros(current_word_models[w].stateTransitions:size())
		priors_per_word[w] = torch.zeros(current_word_models[w].statePriors:size())
		observations_per_word[w] = {}
	end

	return state_transitions_by_word, priors_per_word, observations_per_word
end
