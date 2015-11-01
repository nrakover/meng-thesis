require 'torch'
require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')

require 'loadcaffe'

-- dofile('torch-libs/overfeat-torch/get-net.lua')
dofile('torch-libs/overfeat-torch/load-and-process-img.lua')

local train = torch.load('datasets/stl10/train.t7')
local test = torch.load('datasets/stl10/test.t7')

train.label = train.label:add(-1):byte():add(1)
test.label = test.label:add(-1):byte():add(1)

local dim = 224
local layer_to_extract = 43
function normalizeImage(im)
	local mean_img = torch.FloatTensor(im:size())
	mean_img[{{1},{},{}}] = -123.68
	mean_img[{{2},{},{}}] = -116.779
	mean_img[{{3},{},{}}] = -103.939
	mean_img = mean_img:float()
	return torch.add(im,mean_img):float()
end
-- local net = getOverFeatNet('big')
local net = loadcaffe.load('networks/VGG/VGG_ILSVRC_19_layers_deploy.prototxt', 'networks/VGG/VGG_ILSVRC_19_layers.caffemodel', 'nn')
print('==> Network built\n\n')

local train_features = {data={}}
local test_features = {data={}}

print('==> Generating training features')
for i=1, train.data:size(1) do
	local img = processImage((train.data[{{i},{},{},{}}]):squeeze(), dim)
	-- img:add(-118.380948):div(61.896913)
	img = normalizeImage(img)

	net:forward(img)
	local features = net:get(layer_to_extract).output:clone()
	local feature_vector = nn.View(1):forward(features)

	train_features.data[i] = feature_vector:clone()

	if i%50 == 0 then
		print(100*i/train.data:size(1), '%')
	end
end
print('\n')

print('==> Generating testing features\n\n')
for i=1, test.data:size(1) do
	local img = processImage((test.data[{{i},{},{},{}}]):squeeze(), dim)
	-- img:add(-118.380948):div(61.896913)
	img = normalizeImage(img)

	net:forward(img)
	local features = net:get(layer_to_extract).output:clone()
	local feature_vector = nn.View(1):forward(features)

	test_features.data[i] = feature_vector:clone()

	if i%50 == 0 then
		print(100*i/test.data:size(1), '%')
	end
end
print('\n')


train_features.label = train.label:clone()
test_features.label = test.label:clone()

torch.save('datasets/stl10/VGG-features/train.t7', train_features)
torch.save('datasets/stl10/VGG-features/test.t7', test_features)

print('==> Finished')
