require 'torch'
require 'nn'
require 'loadcaffe'
require 'image'

dofile('/local/nrakover/meng/load-and-process-img.lua')

torch.setdefaulttensortype('torch.FloatTensor')

local NETWORK_PROTOTXT_PATH = '/local/nrakover/meng/networks/VGG/VGG_ILSVRC_19_layers_deploy.prototxt'
local NETWORK_CAFFEMODEL_PATH = '/local/nrakover/meng/networks/VGG/VGG_ILSVRC_19_layers.caffemodel'

local OBJECT_CLASS_PATH_PREFIX = '/local/nrakover/meng/datasets/video-corpus/object_images/'

local function split(inputstr, sep)
    if sep == nil then
            sep = "%s"
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
            t[i] = str
            i = i + 1
    end
    return t
end

local function getImagePaths(class)
	local p = io.popen('find "'..OBJECT_CLASS_PATH_PREFIX..class..'/" -type f')  --Open directory look for files, save data in p. By giving '-type f' as parameter, it returns all files.     
	local img_names = {}
	for file in p:lines() do                      --Loop through all files
		table.insert(img_names, file)
	end
	return img_names
end

local function normalizeImage(im)
	local mean_img = torch.FloatTensor(im:size())
	mean_img[{{1},{},{}}] = -123.68
	mean_img[{{2},{},{}}] = -116.779
	mean_img[{{3},{},{}}] = -103.939
	mean_img = mean_img:float()
	return torch.add(im,mean_img):float()
end

local IMG_DIM = 224
local LAYER_TO_EXTRACT = 43
local function extractFeatures(img, net)
	local processed_img = processImage(img, IMG_DIM)
	local normd_img = normalizeImage(processed_img)
	net:forward(normd_img)
	local features = net:get(LAYER_TO_EXTRACT).output:clone()
	return nn.View(1):forward(features)
end

-- Build network for feature extraction
local net = loadcaffe.load(NETWORK_PROTOTXT_PATH, NETWORK_CAFFEMODEL_PATH, 'nn')
print('==> Network loaded\n\n')


function generateDataset(object_classes)
	-- Dataset as VGG features per class
	local examples_by_class = {}

	-- Extract features for each class
	for j,class in ipairs(object_classes) do
		examples_by_class[class] = {}
		print('\nBuilding examples of class: '..class)
		local image_paths = getImagePaths(class)
		for i,img_path in ipairs(image_paths) do
			local img = image.load(img_path)
			local features = extractFeatures(img[{{1,3},{},{}}], net)
			table.insert(examples_by_class[class], features)

			-- Display progress
			io.write(('  '..(100 * i / #image_paths))..'%', '\r'); io.flush();
		end
	end

	-- Extract examples of background image patched
	examples_by_class['background_image_regions'] = {}
	print('\nBuilding examples of background_image_regions')
	local image_paths = getImagePaths('background_image_regions')
	for i,img_path in ipairs(image_paths) do
		local img = image.load(img_path)
		local features = extractFeatures(img[{{1,3},{},{}}], net)
		table.insert(examples_by_class['background_image_regions'], features)

		-- Display progress
		io.write(('  '..(100 * i / #image_paths))..'%', '\r'); io.flush();
	end

	-- Compile the training set for each class
	for j,class in ipairs(object_classes) do
		local dataset = {data={}, label={}}
		print('Pos class: '..class)
		for i,example in ipairs(examples_by_class[class]) do
			table.insert(dataset.data, example)
			table.insert(dataset.label, 1)
		end

		for neg_class, neg_examples in pairs(examples_by_class) do
			if neg_class ~= class then
				print('Neg class: '..neg_class)
				for i,example in ipairs(neg_examples) do
					table.insert(dataset.data, example)
					table.insert(dataset.label, -1)
				end
			end
		end

		torch.save(class..'_dataset.t7', {data=dataset.data, label=torch.Tensor(dataset.label)})
	end

	print('\n\n==> Finished')
end

generateDataset({'chair', 'trash_bin', 'backpack', 'person'})






