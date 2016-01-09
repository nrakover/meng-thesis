-- require 'svm';
-- dofile('data-to-svmlight.lua' );
dofile('train-test-split.lua' );
dofile('classifiers.lua')


function getClassifier(dataset)
	local training_data = torch.load(dataset);
	
	-- local train_full = t7ToSvmlight(training_data.data, training_data.label);
	-- local classifier = liblinear.train(train_full, '-s 0 -q');
	-- local labels,accuracy,prob = liblinear.predict(train_full, classifier, '-b 1');

	local classifier = trainLinearModel(training_data.data, training_data.label, nil, true)

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
	print("Training 'person' detector")
	local person_classifier = getClassifier('datasets/VOC2007/person_dataset.t7')
	return {emissions={person_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
end

function getCarDetector()
	print("Training 'car' detector")
	local car_classifier = getClassifier('datasets/VOC2007/car_dataset.t7')
	return {emissions={car_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
end


-- ##################
-- ##	ADJECTIVES ##
-- ##################
function getBlackDetector()
	print("Training 'black' detector")
	local black_classifier = getClassifier('datasets/imagenet/attributes_datasets/black.t7')
	return {emissions={black_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
end

function getYellowDetector()
	print("Training 'yellow' detector")
	local yellow_classifier = getClassifier('datasets/imagenet/attributes_datasets/yellow.t7')
	return {emissions={yellow_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
end

function getWhiteDetector()
	print("Training 'white' detector")
	local white_classifier = getClassifier('datasets/imagenet/attributes_datasets/white.t7')
	return {emissions={white_classifier}, transitions=torch.ones(1,1), priors=torch.ones(1)}
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










