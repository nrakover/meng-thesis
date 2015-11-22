require 'image'

function processImage(img_raw, dim)
	
	img_raw = img_raw:mul(255)
	-- local rh = img_raw:size(2)
	-- local rw = img_raw:size(3)
	-- if rh < rw then
	--    rw = math.floor(rw / rh * dim)
	--    rh = dim
	-- else
	--    rh = math.floor(rh / rw * dim)
	--    rw = dim
	-- end
	-- local img_scale = image.scale(img_raw, rw, rh)

	-- local offsetx = 1
	-- local offsety = 1
	-- if rh < rw then
	--    offsetx = offsetx + math.floor((rw-dim)/2)
	-- else
	--    offsety = offsety + math.floor((rh-dim)/2)
	-- end
	-- img = img_scale[{{},{offsety,offsety+dim-1},{offsetx,offsetx+dim-1}}]:int():float()
	local img = image.scale(img_raw, dim, dim):int():float()

	return img
end

function loadAndProcessImage(filename, dim)
	local img_raw = image.load(filename)
	return processImage(img_raw, dim)
end
