
local function permute(tab, count)
	local n = #tab
	for i = 1, count or n do
		local j = math.random(i, n)
		tab[i], tab[j] = tab[j], tab[i]
	end
	return tab
end

local function concatTables(t1, t2)
	local t = {}
	for i = 1, #t1 do
		table.insert(t, t1[i])
	end
	for i = 1, #t2 do
		table.insert(t, t2[i])
	end
	return t
end

local function partialTrainTestSplit(dataset, train_fraction)
	local train_size = math.floor(#dataset.data * train_fraction)
	local indices = {}
	for i = 1,#dataset.data do
		table.insert(indices, i)
	end
	local indices_permutation = permute(indices, train_size)

	local bitmap = torch.zeros(#dataset.data)
	local train_set = {data={}}
	for i = 1, train_size do
		train_set.data[i] = dataset.data[indices_permutation[i]]
		bitmap[indices_permutation[i]] = 1
	end

	bitmap = bitmap:byte()
	if dataset.label ~= nil then
		train_set.label = dataset.label[bitmap]
	end
	if dataset.value ~= nil then
		train_set.value = dataset.value[bitmap]
	end


	local test_set = {data={}}
	for i = train_size+1, #dataset.data do
		table.insert(test_set.data, dataset.data[indices_permutation[i]])
	end

	if dataset.label ~= nil then
		test_set.label = dataset.label[torch.lt(bitmap, 1)]
	end
	if dataset.value ~= nil then
		test_set.value = dataset.value[torch.lt(bitmap, 1)]
	end

	return train_set, test_set
end

function getTrainTestSplit(dataset, train_fraction)
	-- In the regression case
	if dataset.label == nil then
		return partialTrainTestSplit(dataset, train_fraction)
	end

	-- In the classification case, split the positive and negative examples separately
	local pos_dataset = {data={}, label={}}
	local neg_dataset = {data={}, label={}}
	for i = 1, #dataset.data do
		if dataset.label[i] == 1 then
			table.insert(pos_dataset.data, dataset.data[i])
			table.insert(pos_dataset.label, dataset.label[i])
		else
			table.insert(neg_dataset.data, dataset.data[i])
			table.insert(neg_dataset.label, dataset.label[i])
		end
	end
	pos_dataset.label = torch.Tensor(pos_dataset.label)
	neg_dataset.label = torch.Tensor(neg_dataset.label)

	local pos_train, pos_test = partialTrainTestSplit(pos_dataset, train_fraction)
	local neg_train, neg_test = partialTrainTestSplit(neg_dataset, train_fraction)
	local train_set = {data=concatTables(pos_train.data, neg_train.data), label=torch.cat(pos_train.label, neg_train.label)}
	local test_set = {data=concatTables(pos_test.data, neg_test.data), label=torch.cat(pos_test.label, neg_test.label)}

	return train_set, test_set
end
