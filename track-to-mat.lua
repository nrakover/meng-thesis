require 'torch'
local matio = require 'matio'

function trackToMat(track, outfile)
	local track_m = {}
	for i = 1, #track do
		local tensor_i = torch.Tensor(#track[i], 4)
		for r = 1, #track[i] do
			tensor_i[{{r},{}}] = track[i][r]
		end
		track_m['t'..i] = tensor_i
	end
	track_m['nFrames'] = torch.IntTensor({#track})
	matio.save(outfile, track_m)
end