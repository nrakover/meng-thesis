require 'torch'
require 'nn'
dofile('project/pretrained-word-models.lua');

-- word_models = torch.load('learn_open_take1/models_ckpt_10.t7')
-- word_models = torch.load('learn_close_take1/models_ckpt_10.t7')
-- word_models = torch.load('learn_red_blue_take8/models_ckpt_10.t7')
word_models = torch.load('learn_pickup_take58/models_ckpt_10.t7')
-- word_models = torch.load('learn_put_down_take1/models_ckpt_5.t7')
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

sentence = {}
sentence[1] = {word='person', roles={1}}
sentence[2] = {word='pickup', roles={1,2}}
sentence[3] = {word='backpack', roles={2}}
-- sentence[2] = {word='blue', roles={1}}

words_to_filter_by = {}
words_to_filter_by['person'] = true
words_to_filter_by['trash_bin'] = true
words_to_filter_by['chair'] = true
words_to_filter_by['backpack'] = true
words_to_filter_by['book'] = true


VID_DIR = 'datasets/video-corpus/single_track_videos/MVI_0874/'

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

end_time = os.time()
print('Seconds elapsed: '..((end_time - start_time)))
print('==> FINISHED TESTING')
