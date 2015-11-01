require 'torch'

function t7ToSvmlight(data, labels)
	local d = {}
	for i=1, labels:size(1) do
		local record = {}
		record[1] = labels[i]
		
		record[2] = {}
		local sparse_indices = torch.gt(data[i], 0)
		record[2][1] = torch.linspace(1,sparse_indices:size(1), sparse_indices:size(1))[sparse_indices]:int()
		record[2][2] = data[i][sparse_indices]:clone()

		d[i] = record
	end

	return d
end