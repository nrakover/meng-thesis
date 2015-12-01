require 'torch'
require 'nn'
require 'image'
require 'loadcaffe'

dofile('../../load-and-process-img.lua')

local http = require 'socket.http'
local matio = require 'matio'
matio.use_lua_strings = true

torch.setdefaulttensortype('torch.FloatTensor')

local MAT_DIR_PATH = 'attributes_MAT_files/'
local IMG_DIR_PATH = 'images/'
local MAPPING_URL_PREFIX = 'http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid='
local NETWORK_PROTOTXT_PATH = '../../networks/VGG/VGG_ILSVRC_19_layers_deploy.prototxt'
local NETWORK_CAFFEMODEL_PATH = '../../networks/VGG/VGG_ILSVRC_19_layers.caffemodel'

local function dirLookup(dir)
   local p = io.popen('find "'..dir..'" -type f')  --Open directory look for files, save data in p. By giving '-type f' as parameter, it returns all files.     
   local file_names = {}
   for file in p:lines() do                      --Loop through all files
       table.insert(file_names, file)
   end
   return file_names
end

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

local function getImgIDToURLMap(synset_id)
	local mapping_file = http.request(MAPPING_URL_PREFIX..synset_id)
	if mapping_file == nil then return nil end

	local map = {}
	local lns = split(mapping_file, '\n')
	for i=1,#lns-1 do
		local line_mapping = split(lns[i])
		map[line_mapping[1]] = line_mapping[2]
	end
	return map
end

local function getExtension(url)
	local i,j = string.find(string.lower(url), '.jpg')
	if i ~= nil and j == string.len(url) then
		return '.jpg'
	end
	i,j = string.find(string.lower(url), '.jpeg')
	if i ~= nil and j == string.len(url) then
		return '.jpg'
	end
	i,j = string.find(string.lower(url), '.png')
	if i ~= nil and j == string.len(url) then
		return '.png'
	end
	return nil
end

local function downloadImageTensor(url, img_id)
	if url == nil then return nil end
	
	local data = http.request(url)
	if data == nil then return nil end

	local extension = getExtension(url)
	if extension == nil then return nil end

	local filename = IMG_DIR_PATH..img_id..extension
	
	if pcall(
		function()
			local f = assert(io.open(filename, 'w')) -- open in "binary" mode
			f:write(data)
			f:close()
		end
	) then
		-- do nothing, it worked
	else return nil end

	if pcall(function() image.load(filename) end) then
		return image.load(filename)
	else
		return nil
	end
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


local ATTRIBUTES = {
  "black",
  "blue",
  "brown",
  "furry",
  "gray",
  "green",
  "long",
  "metallic",
  "orange",
  "pink",
  "rectangular",
  "red",
  "rough",
  "round",
  "shiny",
  "smooth",
  "spotted",
  "square",
  "striped",
  "vegetation",
  "violet",
  "wet",
  "white",
  "wooden",
  "yellow"
}

-- Tables to hold the attributes datasets
local attr_datasets = {}
for i=1,25 do
	attr_datasets[i] = {data={}, label={}}
end
-- Table to hold the localization datasets
local loc_datasets = {x={data={}, value={}}, y={data={}, value={}}}


function saveDatasets(suffix)
	print('\n===> Saving attribute datasets')

	for i = 1,25 do
		local dataset = {data=attr_datasets[i].data, label=torch.ByteTensor(attr_datasets[i].label)}
		torch.save('attributes_datasets/'..ATTRIBUTES[i]..suffix..'.t7', dataset)
	end

	print('\n===> Saving localization datasets')

	local x_dataset = {data=loc_datasets.x.data, value=torch.FloatTensor(loc_datasets.x.value)}
	torch.save('localization_datasets/x'..suffix..'.t7', x_dataset)

	local y_dataset = {data=loc_datasets.y.data, value=torch.FloatTensor(loc_datasets.y.value)}
	torch.save('localization_datasets/y'..suffix..'.t7', y_dataset)
end


-- Build network for feature extraction
local net = loadcaffe.load(NETWORK_PROTOTXT_PATH, NETWORK_CAFFEMODEL_PATH, 'nn')
print('==> Network loaded\n\n')

-- Iterate over synsets (MAT files)
local synsets = dirLookup(MAT_DIR_PATH)
for synset_indx = 1,#synsets do
	local synset_file = synsets[synset_indx]
	local i,_ = string.find(synset_file, '.attrann.mat')
	local synset_id = string.sub(synset_file, 22, i-1)

	-- Get the map from image ID to image URL
	local synset_map = getImgIDToURLMap(synset_id)

	if synset_map ~= nil then
		-- Get the data
		local attrann = matio.load(synset_file, 'attrann')

		-- Iterate over each image with attributes
		for img_indx = 1,#attrann.images do
			local img_id = attrann.images[img_indx]
			-- Map the image ID to its download URL
			local img_url = synset_map[img_id]

			-- Download image, proceed if successful
			local img = downloadImageTensor(img_url, img_id)
			if img ~= nil then

				-- Get full image features
				local full_img_features = extractFeatures(img, net)

				-- Get the bounding box
				local x1 = attrann.bboxes[img_indx][1].x1[1][1]
				local x2 = attrann.bboxes[img_indx][1].x2[1][1]
				local y1 = attrann.bboxes[img_indx][1].y1[1][1]
				local y2 = attrann.bboxes[img_indx][1].y2[1][1]

				local x1_pix = math.floor(x1*img:size(3))
				local x2_pix = math.floor(x2*img:size(3))
				local y1_pix = math.floor(y1*img:size(2))
				local y2_pix = math.floor(y2*img:size(2))

				-- Add full image to localization dataset
				table.insert(loc_datasets.x.data, full_img_features)
				table.insert(loc_datasets.x.value, (x1+x2)/2)

				table.insert(loc_datasets.y.data, full_img_features)
				table.insert(loc_datasets.y.value, (y1+y2)/2)

				-- Extract image region
				local img_region = image.crop(img, x1_pix, y1_pix, x2_pix, y2_pix)

				-- Extract image region features
				local img_region_features = extractFeatures(img_region, net)

				-- Iterate over attributes
				for attr_indx = 1,#attrann.attributes do
					-- If label is available, put features and label in dataset
					if attrann.labels[img_indx][attr_indx] ~= 0 then
						table.insert(attr_datasets[attr_indx].data, img_region_features)
						table.insert(attr_datasets[attr_indx].label, attrann.labels[img_indx][attr_indx])
					end
				end
			end
		end
	end

	-- Display progress
	io.write(('  '..(100 * synset_indx / #synsets))..'%', '\r'); io.flush();

	-- Save progress
	if synset_indx == math.floor(#synsets/2) then
		saveDatasets('_FIRST_HALF')
	end
end

-- Save datasets
saveDatasets('')

print('\n===> Finished')











