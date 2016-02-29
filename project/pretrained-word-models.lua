require 'torch'

-- require 'svm';
-- dofile('data-to-svmlight.lua' );
dofile('train-test-split.lua' );
dofile('classifiers.lua')

local DETECTORS_PATH = '/local/nrakover/meng/detectors-depot/'


function getClassifier(dataset, num_epochs)
	local training_data = torch.load(dataset);
	
	-- local train_full = t7ToSvmlight(training_data.data, training_data.label);
	-- local classifier = liblinear.train(train_full, '-s 0 -q');
	-- local labels,accuracy,prob = liblinear.predict(train_full, classifier, '-b 1');

	local classifier = trainLinearModel(training_data.data, training_data.label, nil, num_epochs, true)

	return classifier
end

-- train, test = getTrainTestSplit(detector_training_data, 0.9);
-- train_d = t7ToSvmlight(train.data, train.label);
-- test_d = t7ToSvmlight(test.data, test.label);
-- classifier = liblinear.train(train_d, '-s 0 -q');
-- labels,accuracy,prob = liblinear.predict(test_d, classifier, '-b 1');


-- ##################
-- ##	  NOUNS	   ##
-- ##################
function getPersonDetector()
	local person_classifier = nil
	if pcall(function() person_classifier = torch.load(DETECTORS_PATH..'person.t7') end) then
		print("Using cached 'person' detector")
	else
		print("Training 'person' detector")
		person_classifier = getClassifier('datasets/video-corpus/object_images/person_dataset.t7', 5)
		-- person_classifier = getClassifier('datasets/VOC2007/person_dataset.t7', 10)
		torch.save(DETECTORS_PATH..'person.t7', person_classifier)
	end
	return {emissions={person_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
end

function getCarDetector()
	local car_classifier = nil
	if pcall(function() car_classifier = torch.load(DETECTORS_PATH..'car.t7') end) then
		print("Using cached 'car' detector")
	else
		print("Training 'car' detector")
		car_classifier = getClassifier('datasets/VOC2007/car_dataset.t7')
		torch.save(DETECTORS_PATH..'car.t7', car_classifier)
	end
	return {emissions={car_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
end

function getChairDetector()
	local chair_classifier = nil
	if pcall(function() chair_classifier = torch.load(DETECTORS_PATH..'chair.t7') end) then
		print("Using cached 'chair' detector")
	else
		print("Training 'chair' detector")
		chair_classifier = getClassifier('datasets/video-corpus/object_images/chair_dataset.t7', 4)
		-- chair_classifier = getClassifier('datasets/VOC2007/chair_dataset.t7')
		torch.save(DETECTORS_PATH..'chair.t7', chair_classifier)
	end
	return {emissions={chair_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
end

function getTrashbinDetector()
	
	local trash_bin_classifier = nil
	if pcall(function() trash_bin_classifier = torch.load(DETECTORS_PATH..'trash_bin.t7') end) then
		print("Using cached 'trash_bin' detector")
	else
		print("Training 'trash_bin' detector")
		trash_bin_classifier = getClassifier('datasets/video-corpus/object_images/trash_bin_dataset.t7', 3)
		torch.save(DETECTORS_PATH..'trash_bin.t7', trash_bin_classifier)
	end
	return {emissions={trash_bin_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
end

function getBackpackDetector()
	local backpack_classifier = nil
	if pcall(function() backpack_classifier = torch.load(DETECTORS_PATH..'backpack.t7') end) then
		print("Using cached 'backpack' detector")
	else
		print("Training 'backpack' detector")
		backpack_classifier = getClassifier('datasets/video-corpus/object_images/backpack_dataset.t7', 5)
		torch.save(DETECTORS_PATH..'backpack.t7', backpack_classifier)
	end
	return {emissions={backpack_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
end


-- ##################
-- ##	ADJECTIVES ##
-- ##################
function getBlackDetector()
	print("Training 'black' detector")
	local black_classifier = nil
	if pcall(function() black_classifier = torch.load(DETECTORS_PATH..'black.t7') end) == false then
		black_classifier = getClassifier('datasets/imagenet/attributes_datasets/black.t7', 5)
		torch.save(DETECTORS_PATH..'black.t7', black_classifier)
	end
	return {emissions={black_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
end

function getYellowDetector()
	print("Training 'yellow' detector")
	local yellow_classifier = nil
	if pcall(function() yellow_classifier = torch.load(DETECTORS_PATH..'yellow.t7') end) == false then
		yellow_classifier = getClassifier('datasets/imagenet/attributes_datasets/yellow.t7', 5)
		torch.save(DETECTORS_PATH..'yellow.t7', yellow_classifier)
	end
	return {emissions={yellow_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
end

function getWhiteDetector()
	print("Training 'white' detector")
	local white_classifier = nil
	if pcall(function() white_classifier = torch.load(DETECTORS_PATH..'white.t7') end) == false then
		white_classifier = getClassifier('datasets/imagenet/attributes_datasets/white.t7', 5)
		torch.save(DETECTORS_PATH..'white.t7', white_classifier)
	end
	return {emissions={white_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
end

function getBlueDetector()
	print("Training 'blue' detector")
	local blue_classifier = nil
	if pcall(function() blue_classifier = torch.load(DETECTORS_PATH..'blue.t7') end) == false then
		blue_classifier = getClassifier('datasets/imagenet/attributes_datasets/blue.t7', 2)
		torch.save(DETECTORS_PATH..'blue.t7', blue_classifier)
	end
	return {emissions={blue_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
end

function getRedDetector()
	print("Training 'red' detector")
	local red_classifier = nil
	if pcall(function() red_classifier = torch.load(DETECTORS_PATH..'red.t7') end) == false then
		red_classifier = getClassifier('datasets/imagenet/attributes_datasets/red.t7', 3)
		torch.save(DETECTORS_PATH..'red.t7', red_classifier)
	end
	return {emissions={red_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
end

function getGrayDetector()
	print("Training 'gray' detector")
	local gray_classifier = nil
	if pcall(function() gray_classifier = torch.load(DETECTORS_PATH..'gray.t7') end) == false then
		gray_classifier = getClassifier('datasets/imagenet/attributes_datasets/gray.t7', 5)
		torch.save(DETECTORS_PATH..'gray.t7', gray_classifier)
	end
	return {emissions={gray_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
end


-- ##################
-- ##	  VERBS	   ##
-- ##################
-- function getApproachDetector() -- implement
-- 	-- long-range detector
-- 	-- mid-range detector
-- 	-- close-range detector

-- 	-- transition probabilities

-- 	-- prior distribution
-- end










