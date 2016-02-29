dofile('/local/nrakover/meng/project/pretrained-word-models.lua');
dofile('/local/nrakover/meng/project/learner.lua');

local SENTENCES_PATH_PREFIX = '/local/nrakover/meng/datasets/video-corpus/sentences/'

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

local function concatTables(t1, t2)
	local t = {}
	for i = 1, #t1 do
		table.insert(t, t1[i])
	end
	for i = 1, #t2 do
		table.insert(t, t2[i])
	end
	return t
end

local function loadSentences(sentence_set_name)
	local sentences = {}
	local video_names = {}

	local f = assert(io.open(SENTENCES_PATH_PREFIX..sentence_set_name,'r'))
	for line in f:lines() do
		local split_by_tabs = split(line, '\t')
		table.insert(video_names, split_by_tabs[1])
		
		local encoded_sentence = split_by_tabs[2]
		local split_by_commas = split(encoded_sentence, ',')
		local sentence = {}
		for i = 1,#split(split_by_commas[1],' ') do
			local w = split(split_by_commas[1],' ')[i]
			table.insert(sentence, {word=w, roles={1}})
		end
		table.insert(sentence, {word=split_by_commas[2], roles={1,2}})
		for i = 1,#split(split_by_commas[3],' ') do
			local w = split(split_by_commas[3],' ')[i]
			table.insert(sentence, {word=w, roles={2}})
		end
		table.insert(sentences, sentence)
	end

	return sentences, video_names
end

local function getVideos(videos_dir, video_names)
	local videos = {}
	for i = 1, #video_names do
		videos[i] = {detections_path=videos_dir..video_names[i]..'/detections.mat', features_path=videos_dir..video_names[i]..'/features.t7', opticalflow_path=videos_dir..video_names[i]..'/opticalflow.t7'}
	end
	return videos
end

local function initClassifier(num_arguments)
	local NUM_INPUTS = 4096
	if num_arguments == 2 then
		NUM_INPUTS = (4096 + 2 + 2) * num_arguments + 1 + 1
	end
	local classifier = nn.Sequential()
	classifier:add(nn.Linear(NUM_INPUTS, 1)) -- linear regression layer
	classifier:add(nn.Sigmoid()) -- signoid for squeezing into probability
	return classifier
end

local function initTransitionMatrix(n)
	local T = torch.zeros(n,n)
	for i = 1, n-1 do
		T[{{i},{i,i+1}}] = torch.Tensor({0.99, 0.01})
		T[{{i},{}}] = T[{{i},{}}] / T[{{i},{}}]:sum()
	end
	T[{{n},{n}}] = 1
	print(T)
	return T
end

local function initPriors(n)
	local P = torch.zeros(n)
	P[1] = 1
	return P
end

local function initKStateWord(num_arguments, num_states)
	local classifiers = {}
	for i = 1, num_states do
		classifiers[i] = initClassifier(num_arguments)
	end
	return {emissions=classifiers, transitions=initTransitionMatrix(num_states), priors=initPriors(num_states)}
end



local initial_word_models = {}
initial_word_models['person'] = getPersonDetector()
initial_word_models['trash_bin'] = getTrashbinDetector()
initial_word_models['chair'] = getChairDetector()
initial_word_models['backpack'] = getBackpackDetector()

local words_to_learn = {'pickup'}
initial_word_models['pickup'] = initKStateWord(2, 3)

-- local words_to_learn = {'approach'}
-- initial_word_models['approach'] = initKStateWord(2, 3)

words_to_filter_by = {}
words_to_filter_by['person'] = true
words_to_filter_by['trash_bin'] = true
words_to_filter_by['chair'] = true
words_to_filter_by['backpack'] = true

local positive_sentences, positive_example_names = loadSentences('pickup/positive_sentences.txt')
local negative_sentences, negative_example_names = loadSentences('pickup/negative_sentences.txt')
local sentences = concatTables(positive_sentences, negative_sentences)

local example_names = concatTables(positive_example_names, negative_example_names)
local videos = getVideos('/local/nrakover/meng/datasets/video-corpus/single_track_videos/', example_names)
local labels = torch.cat(torch.ones(#positive_sentences), -torch.ones(#negative_sentences))

local start_time = os.time()											-- ############################# CHANGE BACK  (filter_detections --> true)
local learned_word_models = WordLearner:learnWords( '/local/nrakover/meng/learn_pickup_take43/models', words_to_learn, videos, sentences, labels, initial_word_models, 10, true, words_to_filter_by )
local end_time = os.time()

print('Minutes elapsed: '..((end_time - start_time)/60))
