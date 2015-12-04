require 'svm';
dofile('data-to-svmlight.lua' );
dofile('train-test-split.lua' );

-- person_training_data = torch.load('datasets/VOC2007/person_dataset.t7');
car_training_data = torch.load('datasets/VOC2007/car_dataset.t7');
black_training_data = torch.load('datasets/imagenet/attributes_datasets/black_FIRST_HALF.t7');


-- train, test = getTrainTestSplit(detector_training_data, 0.9);
-- train_d = t7ToSvmlight(train.data, train.label);
-- test_d = t7ToSvmlight(test.data, test.label);
-- classifier = liblinear.train(train_d, '-s 0 -q');
-- labels,accuracy,prob = liblinear.predict(test_d, classifier, '-b 1');

-- train_full = t7ToSvmlight(person_training_data.data, person_training_data.label);
-- person_classifier = liblinear.train(train_full, '-s 0 -q');
-- labels,accuracy,prob = liblinear.predict(train_full, person_classifier, '-b 1');

train_full = t7ToSvmlight(car_training_data.data, car_training_data.label);
car_classifier = liblinear.train(train_full, '-s 0 -q');
labels,accuracy,prob = liblinear.predict(train_full, car_classifier, '-b 1');

train_full = t7ToSvmlight(black_training_data.data, black_training_data.label);
black_classifier = liblinear.train(train_full, '-s 0 -q');
labels,accuracy,prob = liblinear.predict(train_full, black_classifier, '-b 1');



word_models = {}
word_models['car'] = {emissions={car_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
word_models['black'] = {emissions={black_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}

sentence = {}
sentence[1] = {word='black', roles={1}}
sentence[2] = {word='car', roles={1}}


dofile('project/sentence-hmm.lua');
dofile('track-to-mat.lua')
sentence = SentenceTracker:new(sentence, 'script_in/black-lambo1.mat', 'script_in/black-lambo1_features.t7', 'script_in/black-lambo1_opticalflow.t7', word_models)
track = sentence:getBestTrack()

trackToMat(track, 'script_out/black-lambo1-BLACK_CAR_TRACK.mat')


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

print('==> FINISHED TESTING')
