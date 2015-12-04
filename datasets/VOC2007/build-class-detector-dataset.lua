require 'torch'
require 'nn'
require 'loadcaffe'
require 'LuaXML'
require 'image'

dofile('../../load-and-process-img.lua')

torch.setdefaulttensortype('torch.FloatTensor')


local IMAGE_SET_PATH = 'ImageSets/Main/'
local IMAGES_PREFIX = 'JPEGImages/'
local ANNOTATIONS_PREFIX = 'XML_Annotations/'

local NETWORK_PROTOTXT_PATH = '../../networks/VGG/VGG_ILSVRC_19_layers_deploy.prototxt'
local NETWORK_CAFFEMODEL_PATH = '../../networks/VGG/VGG_ILSVRC_19_layers.caffemodel'


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

local function getListOfLabeledImageIDs(class)
	local f = io.open(IMAGE_SET_PATH..class..'_trainval.txt')
	local images = {}
	for line in f:lines() do
		table.insert(images, split(line))
	end
	f:close()
	return images
end


function normalizeImage(im)
	local mean_img = torch.FloatTensor(im:size())
	mean_img[{{1},{},{}}] = -123.68
	mean_img[{{2},{},{}}] = -116.779
	mean_img[{{3},{},{}}] = -103.939
	mean_img = mean_img:float()
	return torch.add(im,mean_img):float()
end

local IMG_DIM = 224
local LAYER_TO_EXTRACT = 43
function extractFeatures(img, net)
	local processed_img = processImage(img, IMG_DIM)
	local normd_img = normalizeImage(processed_img)
	net:forward(normd_img)
	local features = net:get(LAYER_TO_EXTRACT).output:clone()
	return nn.View(1):forward(features)
end

-- Build network for feature extraction
local net = loadcaffe.load(NETWORK_PROTOTXT_PATH, NETWORK_CAFFEMODEL_PATH, 'nn')
print('==> Network loaded\n\n')


function generateDataset(target_class, outfile, numPositiveExamples, numNegativeExamples, numRandNegativeSamples)
	-- Dataset as VGG features
	local dataset = {data={}, label={}}

	-- Get the IDs of all files labeled for the desired class
	local imageIDs = getListOfLabeledImageIDs(target_class)

	local pCount = 0
	local nCount = 0
	-- For each labeled image
	for i = 1, #imageIDs do
		-- Terminate when we have enough samples
		if numPositiveExamples ~= nil and pCount >= numPositiveExamples and numNegativeExamples ~= nil and nCount >= numNegativeExamples then break end

		if tonumber(imageIDs[i][2]) == 1 or ((numPositiveExamples == nil or pCount >= numPositiveExamples) and tonumber(imageIDs[i][2]) == -1) then 
			local imID = imageIDs[i][1]

			-- Get the image
			local img = image.load(IMAGES_PREFIX..imID..'.jpg')

			-- Get the annotations
			local annotation = xml.load(ANNOTATIONS_PREFIX..imID..'.xml')

			-- Iterate over the XML tags
			for tagIndx = 1, #annotation do
				local tag = annotation[tagIndx]

				-- If tag is an object
				if tag:find('object') ~= nil then
					-- Get image region
					local bndbox = tag:find('bndbox')
					local x_min = tonumber(bndbox:find('xmin')[1])
					local y_min = tonumber(bndbox:find('ymin')[1])
					local x_max = tonumber(bndbox:find('xmax')[1])
					local y_max = tonumber(bndbox:find('ymax')[1])

					-- local img_region = image.crop(img, x_min, y_min, x_man, y_max)
					local img_region = img[{{}, {y_min,y_max}, {x_min, x_max}}]:clone()

					-- Extract features
					local features = extractFeatures(img_region, net)
					table.insert(dataset.data, features)

					-- Add to dataset with appropriate label
					if tag:find('name')[1] == target_class then
						table.insert(dataset.label, 1)
						pCount = pCount + 1
					else
						table.insert(dataset.label, 0)
						nCount = nCount + 1
					end
				end
			end

			-- Sample random image region when there are no instances of desired object in the image
			if tonumber(imageIDs[i][2]) == -1 then
				local imID = imageIDs[i][1]

				-- Get the image
				local img = image.load(IMAGES_PREFIX..imID..'.jpg')

				-- Get the annotations
				local annotation = xml.load(ANNOTATIONS_PREFIX..imID..'.xml')

				local width = tonumber(annotation:find('size'):find('width')[1])
				local height = tonumber(annotation:find('size'):find('height')[1])

				for sample_num = 1, numRandNegativeSamples do
					local x_min = math.random(width-1)
					local x_max = math.random(x_min+1, width)
					local y_min = math.random(height-1)
					local y_max = math.random(y_min+1, height)

					local img_region = img[{{}, {y_min,y_max}, {x_min, x_max}}]:clone()

					-- Extract features
					local features = extractFeatures(img_region, net)
					table.insert(dataset.data, features)

					table.insert(dataset.label, 0)
					nCount = nCount + 1
				end
			end
		end

		if numPositiveExamples ~= nil then
			print(100 * pCount / numPositiveExamples)
		else
			print( ('Pos: '..pCount)..('   Neg: '..nCount) )
		end
	end

	-- Save dataset
	torch.save(outfile, {data=dataset.data, label=torch.ByteTensor(dataset.label)})
end


generateDataset('car', 'car_dataset.t7', nil, nil, 4)




