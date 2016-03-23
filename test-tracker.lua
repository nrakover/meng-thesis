require 'torch'
require 'nn'
dofile('project/pretrained-word-models.lua');

-- word_models = torch.load('learn_open_take1/models_ckpt_10.t7')
-- word_models = torch.load('learn_close_take1/models_ckpt_10.t7')
-- word_models = torch.load('learn_red_blue_take8/models_ckpt_10.t7')
word_models = torch.load('learn_pickup_take58/models_ckpt_10.t7')
-- word_models = torch.load('learn_approach_take13/models_ckpt_10.t7')
-- word_models = torch.load('learn_leave_take1/models_ckpt_10.t7')

-- word_models = {}
word_models['person'] = getPersonOldCorpusVersionDetector()
-- word_models['person'] = getPersonSkippingVersionDetector()
-- word_models['person'] = getPersonBooksVersionDetector()
word_models['trash_bin'] = getTrashbinDetector()
word_models['chair'] = getChairDetector()
word_models['backpack'] = getBackpackDetector()
word_models['book'] = getBookDetector()
word_models['lamp'] = getLampDetector()

sentence = {}
sentence[1] = {word='person', roles={1}}
sentence[2] = {word='pickup', roles={1,2}}
sentence[3] = {word='chair', roles={2}}
-- sentence[2] = {word='blue', roles={1}}

words_to_filter_by = {}
words_to_filter_by['person'] = true
words_to_filter_by['trash_bin'] = true
words_to_filter_by['chair'] = true
words_to_filter_by['backpack'] = true
words_to_filter_by['book'] = true
words_to_filter_by['lamp'] = true

-- words_to_filter_by['pickup'] = true

VID_DIR = 'datasets/video-corpus/single_track_videos/MVI_0840/'

-- TARGET = 'person'
-- TARGET = 'backpack'
-- TARGET = 'trash_bin'
-- TARGET = 'chair'

-- sentence[1] = {word=TARGET, roles={1}}

dofile('project/sentence-hmm.lua');
dofile('track-to-mat.lua')
start_time = os.time()
local filter_object_proposals = true
sentence_tracker = SentenceTracker:new(sentence, VID_DIR..'detections.mat', VID_DIR..'features.t7', VID_DIR..'opticalflow.t7', word_models, filter_object_proposals, words_to_filter_by)
-- sentence_tracker = SentenceTracker:new(sentence, 'script_in/nico2.mat', 'script_in/nico2_features.t7', 'script_in/nico2_opticalflow.t7', word_models)

track, score = sentence_tracker:getBestTrack()
print('Track score: '..score)

-- sentence_tracker.detectionIndicesPerRole = sentence_tracker:nonMaximalSuppression( sentence_tracker.detectionIndicesPerRole, sentence_tracker.detectionsByFrame, 250, track )

-- track2 = sentence_tracker:getBestTrack()


-- full_track , _, detection_indices_per_role = sentence_tracker:filterDetections( sentence_tracker.detectionsByFrame, sentence_tracker.detectionFeatures, sentence_tracker.detectionsOptFlow, word_models, words_to_filter_by, 4 )

-- track = {}
-- for fIndx = 1, #full_track do
-- 	track[fIndx] = {}
-- 	for i = 1, 50 do
-- 		track[fIndx][i] = full_track[fIndx][detection_indices_per_role[fIndx][1][i]]
-- 	end
-- end

-- for fIndx = 1, #track do
-- 	print(#track[fIndx])
-- 	-- track[fIndx] = {track[fIndx][5], track[fIndx][6], track[fIndx][7], track[fIndx][8]}
-- end

trackToMat(track, 'script_out/LAST_TRACK.mat')
-- trackToMat(track, VID_DIR..TARGET..'.mat')
-- if track2 ~= nil then
-- 	trackToMat(track2, 'script_out/LAST_TRACK_2.mat')
-- 	trackToMat(track2, VID_DIR..TARGET..'2.mat')
-- end

dofile('visualization/visualize-track.lua')
visualizeTrackMAT('script_out/LAST_TRACK.mat', VID_DIR..'detections.mat', VID_DIR..'video.avi')

-- state_transitions_by_word, priors_per_word, observations_per_word, ll = sentence_tracker:partialEStep({'pickup'})
-- print(ll)
-- print(state_transitions_by_word['pickup'])


-- require 'nn';
-- torch.setdefaulttensortype('torch.FloatTensor');
-- require 'loadcaffe';
-- dofile('load-and-process-img.lua');

-- function normalizeImage(im)
-- 	local mean_img = torch.FloatTensor(im:size())
-- 	mean_img[{{1},{},{}}] = -123.68
-- 	mean_img[{{2},{},{}}] = -116.779
-- 	mean_img[{{3},{},{}}] = -103.939
-- 	mean_img = mean_img:float()
-- 	return torch.add(im,mean_img):float()
-- end

-- IMG_DIM = 224
-- LAYER_TO_EXTRACT = 43
-- function extractFeatures(img, net)
-- 	local processed_img = processImage(img, IMG_DIM)
-- 	local normd_img = normalizeImage(processed_img)
-- 	net:forward(normd_img)
-- 	local features = net:get(LAYER_TO_EXTRACT).output:clone()
-- 	return nn.View(1):forward(features)
-- end

-- net = loadcaffe.load('networks/VGG/VGG_ILSVRC_19_layers_deploy.prototxt', 'networks/VGG/VGG_ILSVRC_19_layers.caffemodel', 'nn');

-- require 'ffmpeg';
-- v = ffmpeg.Video{path='script_in/nico_small.avi', fps=30, length=0.3, width=480, height=270};
-- frames = v:totensor(1,1,8);
-- f1 = frames[1]

-- goodRegion = f1[{{},{30,240},{5,70}}]:clone()
-- badRegion = f1[{{},{130,260},{225,425}}]:clone()
-- allRegion = f1[{{},{5,250},{5,460}}]:clone()

-- goodFeatures = extractFeatures(goodRegion, net)
-- badFeatures = extractFeatures(badRegion, net)
-- allFeatures = extractFeatures(allRegion, net)

-- test_d = t7ToSvmlight({goodFeatures, badFeatures, allFeatures}, torch.ByteTensor(3))
-- labels,accuracy,prob = liblinear.predict(test_d, classifier, '-b 1');
end_time = os.time()
print('Seconds elapsed: '..((end_time - start_time)))
print('==> FINISHED TESTING')
