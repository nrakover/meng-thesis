require 'torch'

DATASETS_PATH_PREFIX = 'attributes_datasets/'

function combineDatasets(attribute)
	d1 = torch.load(DATASETS_PATH_PREFIX..attribute..'_FIRST_HALF.t7')
	d2 = torch.load(DATASETS_PATH_PREFIX..attribute..'_SECOND_HALF.t7')

	for i = 1, #d2.data do
		table.insert(d1.data, d2.data[i])
	end

	d1.label = torch.cat(d1.label, d2.label, 1)

	torch.save(DATASETS_PATH_PREFIX..attribute..'.t7', d1)
end


combineDatasets('black')
combineDatasets('yellow')
combineDatasets('blue')
combineDatasets('green')
combineDatasets('gray')
combineDatasets('red')
combineDatasets('white')
