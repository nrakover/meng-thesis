require 'torch'
local matio = require 'matio'

function trackToMat(track, outfile)
	track_m = {}
	for i = 1, #track do
		track_m['t'..i] = track[i]
	end
	track_m['nFrames'] = torch.IntTensor({#track})
	matio.save(outfile, track_m)
end