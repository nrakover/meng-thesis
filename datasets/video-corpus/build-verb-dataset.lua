require 'torch'
require 'ffmpeg'
require 'liuflow'

local ANNOTATIONS_DIR = ''
local DATASETS_DIR = ''

local VIDEO_WIDTH = 640
local VIDEO_HEIGHT = 480

function build2RoleVerbDataset(verb, num_states)
	local examples = getPositiveExamples(verb)
	if examples == nil then print("annotations file is corrupted or doesn't exist"); return end
	-- Iterate over positive example videos
	for i,e in ipairs(examples) do
		local vid_ID = e.id

		local vid_fps = nil -- TODO
		local vid_length = nil -- TODO

		local video = ffmpeg.Video{path=DATASETS_DIR..vid_ID..'.mov', fps=vid_fps, length=vid_length, width=VIDEO_WIDTH, height=VIDEO_HEIGHT}
		local frames = video:totensor(1,1,12)

		-- Iterate over frames / verb state
		for f = 1, frames:size(1) do
			local frame = frames[f]:clone()
			
			-- Iterate over instance of verb
			for j,verb_instance in ipairs(e.instances) do

				-- Iterate over 1st role detections

					-- Iterate over 2nd role detections

			end
		end
	end

	-- Iterate over negative example videos

		-- Iterate over verb states

			-- Randomly select frame

			-- Randomly select 1st role detection

			-- Randomly select 2nd role detection
end



-- Utilities

local function getPositiveExamples(verb)
	local filename = ANNOTATIONS_DIR..verb..'.txt'
	local lines = nil
	if pcall(
		function()
			local f = assert(io.open(filename, 'r')) -- open in "binary" mode
			lines = split(f:read(), '\n')
			f:close()
		end
	) then
		-- do nothing, it worked
	else return nil end

	local examples = {}
	for i = 1, #lines do
		local line = lines[i]
		local line_by_tab = split(line, '\t')
		local vid_ID = line_by_tab[1]
		
		local instances_by_delimiter = split(line_by_tab[2], '::')
		table.insert(examples, {id=vid_ID, instances=instances_by_delimiter})
	end

	return examples
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