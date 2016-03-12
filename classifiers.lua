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
function trainLinearModel( examples, labels, weights, max_epochs, learning_rate, verbose, adagrad )
	-- Define model
	local model = nn.Sequential()
	local num_inputs = examples[1]:size(1)
	model:add(nn.Linear(num_inputs, 1)) -- linear regression layer
	model:add(nn.Sigmoid()) -- signoid for squeezing into probability

	return doGradientDescentOnModel( model, examples, labels, weights, max_epochs, learning_rate, verbose, adagrad )
end

function doGradientDescentOnModel( model, examples, labels, weights, max_epochs, learning_rate, verbose, adagrad )
	verbose = verbose or false
	adagrad = adagrad or false

	-- If not provided, the starting learning rate is 0.01
	learning_rate = learning_rate or 0.01

	-- If not provided, weigh each example by 1
	weights = weights or torch.ones(#examples)

	-- Squeeze inputs
	for i = 1, #examples do
		examples[i] = torch.squeeze(examples[i]):double()
	end

	-- Cannonicalize labels
	labels[torch.ne(labels, 1)] = 0

	if verbose then
		print('# positive examples: '..torch.sum(torch.eq(labels, 1)))
		print('# negative examples: '..(labels:size(1)-torch.sum(torch.eq(labels, 1))) )

		print('total positive weight: '..torch.sum(weights[torch.eq(labels, 1)]))
		print('total negative weight: '..torch.sum(weights[torch.ne(labels, 1)]))
	end

	-- Define training criterion
	local criterion = nn.BCECriterion()

	-- Learn parameters
	local indices = {}
	for i = 1,#examples do
		table.insert(indices, i)
	end

	local max_epochs = max_epochs or 10
	local prev_err = 0
	local exit_threshold = 1e-8
	local historical_gradient_w = torch.zeros(model.modules[1].weight:size())
	local historical_gradient_b = torch.zeros(1)
	for epoch = 1, max_epochs do
		local lr = learning_rate / epoch
		local total_err = 0
		local total_weight = 0

		local indices_permutation = permute(indices) -- randomize examples for robustness
		for i = 1, #examples do
			local indx = indices_permutation[i]

			local x = examples[indx]:clone()
			local target = labels[indx]
			local w = weights[indx]

			local e = 0
			if adagrad then
				e, historical_gradient_w, historical_gradient_b = gradUpdateWithAdagrad(model, x, torch.DoubleTensor({target}), criterion, w, learning_rate, historical_gradient_w, historical_gradient_b)
			else
				e = gradUpdate(model, x, torch.DoubleTensor({target}), criterion, w, lr)
			end
			total_err = total_err + w*e
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
	if verbose then
		local acc = scoreTestSet( model, examples, labels:double(), weights )
		print('Training set accuracy = '..acc)
	end

	return model
end

function gradUpdate(mlp, x, target, criterion, weight, learning_rate)
	local pred = mlp:forward(x)
	local err = criterion:forward(pred, target)
	local gradCriterion = criterion:backward(pred, target)
	mlp:zeroGradParameters()
	mlp:backward(x, gradCriterion)
	mlp:updateParameters(weight * learning_rate)

	return math.abs(err)
end


local SMOOTH_FACTOR = 1e-6
function gradUpdateWithAdagrad(mlp, x, target, criterion, weight, learning_rate, historical_gradient_w, historical_gradient_b)
	local pred = mlp:forward(x)
	local err = criterion:forward(pred, target)
	local gradCriterion = criterion:backward(pred, target)
	mlp:zeroGradParameters()
	mlp:backward(x, gradCriterion)

	local _, gradParams = mlp:parameters()
	-- Update historical gradients
	historical_gradient_w:add( torch.pow(gradParams[1], 2) )
	historical_gradient_b:add( torch.pow(gradParams[2], 2) )
	-- Adjust gradients according to history
	gradParams[1]:cdiv( torch.sqrt(historical_gradient_w) + SMOOTH_FACTOR )
	gradParams[2]:cdiv( torch.sqrt(historical_gradient_b) + SMOOTH_FACTOR )

	-- Update params
	mlp:updateParameters(weight * learning_rate)

	return math.abs(err), historical_gradient_w, historical_gradient_b
end


-- ###############
-- ##	Eval 	##
-- ###############
function predictLabels( model, examples )
	local predictions = {}
	for i,x in ipairs(examples) do
		local score = model:forward(torch.squeeze(x):double())[1]
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
		if (predictions[i] == 1 and targets[i] == 1) or (predictions[i] ~= 1 and targets[i] ~= 1) then
			true_preds = true_preds + weights[i]
		end
		total_weight = total_weight + weights[i]
	end
	return true_preds / total_weight
end


