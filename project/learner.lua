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
	local margin_list = {}

	-- E-Step: calculate posteriors
	local state_transitions_by_word, priors_per_word, observations_per_word, pos_ll, neg_ll = self:EStep( 0, words_to_learn, videos, sentences, labels, current_word_models, filter_detections, words_to_filter_by )
	-- Report starting loglikelihood
	print(('\n\nInitial likelihoods'))
	print('Positive examples avgerage loglikelihood: '..pos_ll)
	print('Negative examples avgerage loglikelihood: '..neg_ll)
	print('Average margin: '..(pos_ll - neg_ll))
	print('=================================================')
	table.insert(loglikelihood_list, {pos_ll, neg_ll})
	table.insert(margin_list, (pos_ll - neg_ll))


	-- Perform iterations of EM
	for iter = 1, maxIters do
		local start_time = os.time()

		print(('\n\nIteration '..iter))
		print('=================================================')

		-- M-Step: re-estimate word models
		current_word_models = self:MStep( current_word_models, words_to_learn, videos, labels, state_transitions_by_word, priors_per_word, observations_per_word, iter )

		-- Checkpoint
		torch.save(output_name..('_ckpt_'..iter)..'.t7', current_word_models)

		-- E-Step: calculate posteriors
		state_transitions_by_word, priors_per_word, observations_per_word, pos_ll, neg_ll = self:EStep( iter, words_to_learn, videos, sentences, labels, current_word_models, filter_detections, words_to_filter_by )

		-- Report loglikelihood of current models
		print('=================================================')
		print('Positive examples avgerage loglikelihood: '..pos_ll)
		print('Negative examples avgerage loglikelihood: '..neg_ll)
		print('Average margin: '..(pos_ll - neg_ll))
		table.insert(loglikelihood_list, {pos_ll, neg_ll})
		table.insert(margin_list, (pos_ll - neg_ll))

		local end_time = os.time()
		print('=================================================')
		print('Iteration time (minutes): '..((end_time - start_time)/60))
		print('=================================================')
	end

	print('\n=================================================')
	print('Loglikelihood sequence:')
	print(loglikelihood_list)
	print('=================================================\n\n')

	print('\n=================================================')
	print('Margin sequence:')
	print(margin_list)
	print('=================================================\n\n')

	
	return current_word_models
end


function WordLearner:EStep( iter, words_to_learn, videos, sentences, labels, current_word_models, filter_detections, words_to_filter_by )
	local state_transitions_by_word, priors_per_word, observations_per_word = self:initSummaryStatistics( words_to_learn, current_word_models )
	local pos_examples_loglikelihood = 0
	local neg_examples_loglikelihood = 0
	local num_pos_examples = 0
	local num_neg_examples = 0

	-- Iterate over examples
	for i = 1, #videos do
		-- print(videos[i].features_path)

		-- Instantiate sentence tracker
		local sentence = sentences[i]
		local video = videos[i]
		local sentence_tracker = SentenceTracker:new(sentence, video.detections_path, video.features_path, video.opticalflow_path, current_word_models, filter_detections, words_to_filter_by)

		-- Get posteriors for the example
		local trans, priors, obs, ll = sentence_tracker:partialEStep( words_to_learn )


		-- Aggregate
		if ll > math.log(0) then -- skip if the likelihood of the example is 0
			local sentence_likelihood_weight = 1 -- math.exp(ll * (1/1000))
			for j = 1, #words_to_learn do
				local w = words_to_learn[j]
				if labels[i] == 1 then
					state_transitions_by_word[w] = state_transitions_by_word[w] + (trans[w] * sentence_likelihood_weight)
					priors_per_word[w] = priors_per_word[w] + (priors[w] * sentence_likelihood_weight)
				else
					-- subtract counts for negative examples
					state_transitions_by_word[w] = state_transitions_by_word[w] - (trans[w] * sentence_likelihood_weight)
					priors_per_word[w] = priors_per_word[w] - (priors[w] * sentence_likelihood_weight)
				end

				-- Compile the observation examples for each word state
				for state = 1, priors[w]:size(1) do
					for k,v in pairs(obs[w][state]) do
						if v.weight * sentence_likelihood_weight > 0 then -- only use examples with non-zero weight
							table.insert(observations_per_word[w][state].examples, v.example)
							table.insert(observations_per_word[w][state].labels, labels[i])
							table.insert(observations_per_word[w][state].weights, v.weight * sentence_likelihood_weight)
						end
					end
				end
			end

			-- Accumulate loglikelihood
			if labels[i] == 1 then
				pos_examples_loglikelihood = pos_examples_loglikelihood + ll
				num_pos_examples = num_pos_examples + 1
			else
				neg_examples_loglikelihood = neg_examples_loglikelihood + ll
				num_neg_examples = num_neg_examples + 1
			end
		else
			if labels[i] == 1 then
				print('==> positive example scored -inf')
			else
				print('==> negative example scored -inf')
			end
			print(sentence)
			print(video.detections_path)
		end

		-- Display progress
		io.write('\tE-step: '..(100 * i / #videos)..'%', '\r'); io.flush();
		-- print('==> done with video '..i)
	end

	-- Edge case when all negative examples receive a likelihood of 0
	if num_neg_examples == 0 then
		num_neg_examples = 1
		neg_examples_loglikelihood = math.log(0)
	end

	return state_transitions_by_word, priors_per_word, observations_per_word, pos_examples_loglikelihood/num_pos_examples, neg_examples_loglikelihood/num_neg_examples
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
		for p = 1, new_state_transitions:size(1)-1 do
			local transition_counts_from_p = state_transition_counts[{{p},{}}]
			-- Offset all counts if there are negative counts
			if transition_counts_from_p:min() < 0 then
				transition_counts_from_p = transition_counts_from_p - transition_counts_from_p:min()
			end
			-- Normalize
			local total_mass = transition_counts_from_p:sum()
			new_state_transitions[{{p},{}}] = (transition_counts_from_p + 0.01) / (total_mass + 0.01*transition_counts_from_p:size(1)) -- smooth by 0.01

			-- Clip probability outside the band diagonal
			local tmp = torch.zeros(1,new_state_transitions:size(1))
			tmp[{{1},{p,p+1}}] = new_state_transitions[{{p},{p,p+1}}]
			new_state_transitions[{{p},{}}] = tmp / tmp:sum()
		end
		-- Last state has a trivial fixed distribution (self-loop)
		new_state_transitions[{{new_state_transitions:size(1)},{}}] = torch.zeros(1,new_state_transitions:size(1))
		new_state_transitions[{{new_state_transitions:size(1)},{new_state_transitions:size(1)}}] = 1
		
		-- new_state_transitions = current_word_models[w].transitions  -- Freeze Transition Matrix
		print(new_state_transitions)

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
			new_emissions_models[state] = doGradientDescentOnModel( current_word_models[w].emissions[state], observations_per_word[w][state].examples, torch.Tensor(observations_per_word[w][state].labels), torch.Tensor(observations_per_word[w][state].weights), math.min(2+iter, 8), 0.05, true, true )
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
