require 'torch'

dofile('/local/nrakover/meng/project/sentence-hmm.lua')
dofile('/local/nrakover/meng/classifiers.lua')

WordLearner = {}

function WordLearner:learnWords( output_name, words_to_learn, videos, sentences, labels, initial_word_models, max_iterations, filter_detections, words_to_filter_by )

	print('\nLearning words:')
	print(words_to_learn)
	print('\n')

	local current_word_models = initial_word_models

	local maxIters = max_iterations or 10

	local loglikelihood_list = {}

	-- Perform iterations of EM
	for iter = 1, maxIters do

		-- E-Step: calculate posteriors
		local state_transitions_by_word, priors_per_word, observations_per_word, pos_ll, neg_ll = self:EStep( words_to_learn, videos, sentences, labels, current_word_models, filter_detections, words_to_filter_by )

		-- Report loglikelihood of current models
		print(('\n\nIteration '..iter))
		print('Positive examples loglikelihood: '..pos_ll)
		print('Negative examples loglikelihood: '..neg_ll)
		print('=================================================')
		table.insert(loglikelihood_list, {pos_ll, neg_ll})

		-- M-Step: re-estimate word models
		current_word_models = self:MStep( current_word_models, words_to_learn, videos, labels, state_transitions_by_word, priors_per_word, observations_per_word, iter )

		-- Checkpoint
		torch.save(output_name..('_ckpt_'..iter)..'.t7', current_word_models)
	end

	print('\n=================================================')
	print('Loglikelihood sequence:')
	print(loglikelihood_list)
	print('=================================================\n\n')

	return current_word_models
end


function WordLearner:EStep( words_to_learn, videos, sentences, labels, current_word_models, filter_detections, words_to_filter_by )
	local state_transitions_by_word, priors_per_word, observations_per_word = self:initSummaryStatistics( words_to_learn, current_word_models )
	local pos_examples_loglikelihood = 0
	local neg_examples_loglikelihood = 0

	-- Iterate over examples
	for i = 1, #videos do
		-- Instantiate sentence tracker
		local sentence = sentences[i]
		local video = videos[i]
		local sentence_tracker = SentenceTracker:new(sentence, video.detections_path, video.features_path, video.opticalflow_path, current_word_models, filter_detections, words_to_filter_by)

		-- Get posteriors for the example
		local trans, priors, obs, ll = sentence_tracker:partialEStep( words_to_learn )

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
			for state = 1, priors[w]:size(1) do
				for k,v in pairs(obs[w][state]) do
					table.insert(observations_per_word[w][state].examples, v.example)
					table.insert(observations_per_word[w][state].labels, labels[i])
					table.insert(observations_per_word[w][state].weights, v.weight)
				end
			end
		end

		-- Accumulate loglikelihood
		if labels[i] == 1 then
			pos_examples_loglikelihood = pos_examples_loglikelihood + ll
		else
			neg_examples_loglikelihood = neg_examples_loglikelihood + ll
		end

		-- Display progress
		io.write('\tE-step: '..(100 * i / #videos)..'%', '\r'); io.flush();
		-- print('==> done with video '..i)
	end

	return state_transitions_by_word, priors_per_word, observations_per_word, pos_examples_loglikelihood, neg_examples_loglikelihood
end

function WordLearner:MStep( current_word_models, words_to_learn, videos, labels, state_transitions_by_word, priors_per_word, observations_per_word, iter )
	local new_word_models = {}

	-- Use old models, then we overwrite the ones we learn
	for w,model in pairs(current_word_models) do
		new_word_models[w] = model
	end

	-- Iterate over words to learn
	for _,w in ipairs(words_to_learn) do

		print('estimating: '..w)

		-- Compute the transition probabilities
		-- ####################################
		local state_transition_counts = state_transitions_by_word[w]
		local new_state_transitions = torch.FloatTensor(state_transition_counts:size())
		for p = 1, new_state_transitions:size(1) do
			local transition_counts_from_p = state_transition_counts[{{p},{}}]
			-- Offset all counts if there are negative counts
			if transition_counts_from_p:min() < 0 then
				transition_counts_from_p = transition_counts_from_p - transition_counts_from_p:min()
			end
			-- Normalize
			local total_mass = transition_counts_from_p:sum()
			new_state_transitions[{{p},{}}] = (transition_counts_from_p + 0.01) / (total_mass + 0.01*transition_counts_from_p:size(1)) -- smooth by 0.01
		end

		-- Compute the state priors
		-- ####################################
		local state_priors_counts = priors_per_word[w]
		-- Offset counts if there are negative counts
		if state_priors_counts:min() < 0 then
			state_priors_counts = state_priors_counts - state_priors_counts:min()
		end
		-- Normalize
		local new_state_priors = (state_priors_counts + 0.01) / (state_priors_counts:sum() + 0.01*state_priors_counts:size(1)) -- smooth by 0.01

		-- Compute the emissions models
		-- ####################################
		local new_emissions_models = {}
		for state = 1, state_priors_counts:size(1) do
			new_emissions_models[state] = trainLinearModel(observations_per_word[w][state].examples, torch.Tensor(observations_per_word[w][state].labels), observations_per_word[w][state].weights, math.min(2+iter, 10), true)
		end


		-- Update the model parameters
		new_word_models[w] = {}
		new_word_models[w].transitions = new_state_transitions
		new_word_models[w].priors = new_state_priors
		new_word_models[w].emissions = new_emissions_models
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
