require 'torch'
require 'nn'

-- ###############
-- ##	Util 	##
-- ###############
local function permute(tab, count)
	local n = #tab
	for i = 1, count or n do
		local j = math.random(i, n)
		tab[i], tab[j] = tab[j], tab[i]
	end
	return tab
end

-- ###############
-- ##	Train 	##
-- ###############
function trainLinearModel( examples, labels, weights, verbose )
	verbose = verbose or false

	-- If not provided, weigh each example by 1
	weights = weights or torch.ones(#examples)

	-- Squeeze inputs
	for i = 1, #examples do
		examples[i] = torch.squeeze(examples[i]):double()
	end

	-- Cannonicalize labels
	labels[torch.ne(labels, 1)] = 0

	print('# positive examples: '..torch.sum(torch.eq(labels, 1)))
	print('# negative examples: '..(labels:size(1)-torch.sum(torch.eq(labels, 1))) )

	-- Define model
	local model = nn.Sequential()
	local num_inputs = examples[1]:size(1)
	model:add(nn.Linear(num_inputs, 1)) -- linear regression layer
	model:add(nn.Sigmoid()) -- signoid for squeezing into probability

	-- Define training criterion
	local criterion = nn.BCECriterion()

	-- Learn parameters
	local indices = {}
	for i = 1,#examples do
		table.insert(indices, i)
	end

	local max_epochs = 10
	local prev_err = 0
	local exit_threshold = 1e-8
	for epoch = 1, max_epochs do
		local lr = 0.01 / epoch
		local total_err = 0
		local total_weight = 0

		local indices_permutation = permute(indices) -- randomize examples for robustness
		for i = 1, #examples do
			local indx = indices_permutation[i]

			local x = examples[indx]:clone()
			local target = labels[indx]
			local w = weights[indx]

			total_err = total_err + w*gradUpdate(model, x, torch.DoubleTensor({target}), criterion, w, lr)
			total_weight = total_weight + w
		end

		local avg_err = total_err / total_weight
		if math.abs(avg_err - prev_err) < exit_threshold then break end

		if verbose then 
			print(('Epoch '..epoch)..' avg error = '..avg_err) 
		end
		prev_err = avg_err
	end

	-- Print out accuracy on training set
	local acc = scoreTestSet( model, examples, labels:double(), weights )
	print('Training set accuracy = '..acc)

	return model
end

function gradUpdate(mlp, x, target, criterion, weight, learning_rate)
	local pred = mlp:forward(x)
	local err = criterion:forward(pred, target)
	local gradCriterion = criterion:backward(pred, target)
	mlp:zeroGradParameters()
	mlp:backward(x, gradCriterion)
	mlp:updateParameters(weight * learning_rate)

	return err^2
end


-- ###############
-- ##	Eval 	##
-- ###############
function predictLabels( model, examples )
	local predictions = {}
	for i,x in ipairs(examples) do
		local score = model:forward(x)[1]
		if score >= 0.5 then
			predictions[i] = 1
		else
			predictions[i] = 0
		end
	end
	return torch.Tensor(predictions)
end

function scoreTestSet( model, examples, targets, weights )
	weights = weights or torch.ones(#examples)

	local predictions = predictLabels( model, examples )
	local true_preds = 0
	local total_weight = 0
	for i = 1, #examples do
		if predictions[i] == targets[i] then
			true_preds = true_preds + weights[i]
		end
		total_weight = total_weight + weights[i]
	end
	return true_preds / total_weight
end


