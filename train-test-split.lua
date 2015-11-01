
local function permute(tab, count)
	local n = #tab
	for i = 1, count or n do
		local j = math.random(i, n)
		tab[i], tab[j] = tab[j], tab[i]
	end
	return tab
end

function getTrainTestSplit(dataset, train_fraction)
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