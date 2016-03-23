require 'torch'
dofile('/local/nrakover/meng/classifiers.lua')
dofile('/local/nrakover/meng/train-test-split.lua')


local function scaleFeatures( data, scale_factor )
	local scaled_data = {}
	for i = 1, #data do
		scaled_data[i] = data[i] * scale_factor
	end
	return scaled_data
end

local function appendRandomOptflow( data, zero_all )
	local new_data = {}
	for i = 1, #data do
		if zero_all then
			new_data[i] = torch.cat(data[i]:clone(), torch.FloatTensor({0,0,0}), 1)
		else
			local flow = (torch.rand(2) - 0.5) * 30
			local magnitude = torch.norm(flow)
			local normalized_flow = flow / magnitude
			new_data[i] = torch.cat(torch.cat(data[i]:clone(), normalized_flow:float(), 1), torch.FloatTensor({magnitude}), 1)
		end
	end
	return new_data
end

local function trainAndTest(train_data, test_data, num_epochs, adagrad, append_rand_flow)
	local train_X = nil
	local test_X = nil

	if append_rand_flow then
		train_X = scaleFeatures(train_data.data, 0.0001)
		test_X = scaleFeatures(test_data.data, 0.0001)

		train_X = appendRandomOptflow(train_X, true)
		test_X = appendRandomOptflow(test_X)

		train_X = scaleFeatures(train_X, 1000)
		test_X = scaleFeatures(test_X, 1000)
	else
		train_X = scaleFeatures(train_data.data, 0.1)
		test_X = scaleFeatures(test_data.data, 0.1)
	end

	local model = trainLinearModel(train_X, train_data.label, nil, num_epochs, 0.01, true, adagrad)
	
	if append_rand_flow then
		model:get(1).weight[{{1},{4097,4099}}] = 0
		print(model:get(1).weight[{{1},{4097,4099}}])
	end

	local acc = scoreTestSet(model, test_X, test_data.label)
	print('------------------------------------')
	print('Test accuracy = '..acc)
	print('------------------------------------\n')
end


local function testPersonClassifiers()
	-- local VOC_person_data = torch.load('datasets/VOC2007/person_dataset.t7')
	local video_corp_person_data = torch.load('datasets/video-corpus/object_images/person_dataset.t7')

	local person_train, person_test = getTrainTestSplit(video_corp_person_data, 0.8)

	print('\n\n')
	print('====================================')
	print('====          PERSON            ====')
	print('====================================')

	local epochs = {5, 10}
	for i,n in ipairs(epochs) do
		-- print('Video corpus data: '..(n..' epochs'))
		-- trainAndTest(person_train, person_test, n)
		print('Video corpus data: '..(n..' epochs with ADAGRAD'))
		trainAndTest(person_train, person_test, n, true)
		print('Video corpus data: '..(n..' epochs with appended random flow'))
		trainAndTest(person_train, person_test, n, true, true)
	end

	-- print('VOC2007 data: 10 epochs')
	-- trainAndTest(VOC_person_data, person_test, 10)
	-- print('VOC2007 data: 15 epochs')
	-- trainAndTest(VOC_person_data, person_test, 15)
	-- print('VOC2007 data: 20 epochs')
	-- trainAndTest(VOC_person_data, person_test, 20)
end

local function testBackpackClassifiers()
	local video_corp_backpack_data = torch.load('datasets/video-corpus/object_images/backpack_dataset.t7')

	local backpack_train, backpack_test = getTrainTestSplit(video_corp_backpack_data, 0.8)

	print('\n\n')
	print('====================================')
	print('====         BACKPACK           ====')
	print('====================================')

	local epochs = {5, 10}
	for i,n in ipairs(epochs) do
		-- print('Video corpus data: '..(n..' epochs'))
		-- trainAndTest(backpack_train, backpack_test, n)
		print('Video corpus data: '..(n..' epochs with ADAGRAD'))
		trainAndTest(backpack_train, backpack_test, n, true)
		print('Video corpus data: '..(n..' epochs with appended random flow'))
		trainAndTest(backpack_train, backpack_test, n, true, true)
	end

end

local function testChairClassifiers()
	local video_corp_chair_data = torch.load('datasets/video-corpus/object_images/chair_dataset.t7')

	local chair_train, chair_test = getTrainTestSplit(video_corp_chair_data, 0.8)

	print('\n\n')
	print('====================================')
	print('====           CHAIR            ====')
	print('====================================')

	local epochs = {5, 10}
	for i,n in ipairs(epochs) do
		-- print('Video corpus data: '..(n..' epochs'))
		-- trainAndTest(chair_train, chair_test, n)
		print('Video corpus data: '..(n..' epochs with ADAGRAD'))
		trainAndTest(chair_train, chair_test, n, true)
		print('Video corpus data: '..(n..' epochs with appended random flow'))
		trainAndTest(chair_train, chair_test, n, true, true)
	end	
end

local function testTrashbinClassifiers()
	local video_corp_trashbin_data = torch.load('datasets/video-corpus/object_images/trash_bin_dataset.t7')

	local trashbin_train, trashbin_test = getTrainTestSplit(video_corp_trashbin_data, 0.8)

	print('\n\n')
	print('====================================')
	print('====         TRASH_BIN          ====')
	print('====================================')

	local epochs = {5, 10}
	for i,n in ipairs(epochs) do
		-- print('Video corpus data: '..(n..' epochs'))
		-- trainAndTest(trashbin_train, trashbin_test, n)
		print('Video corpus data: '..(n..' epochs with ADAGRAD'))
		trainAndTest(trashbin_train, trashbin_test, n, true)
		print('Video corpus data: '..(n..' epochs with appended random flow'))
		trainAndTest(trashbin_train, trashbin_test, n, true, true)
	end
end

local function testBlackClassifiers()
	local color_black_data = torch.load('datasets/imagenet/attributes_datasets/black.t7')

	local color_black_train, color_black_test = getTrainTestSplit(color_black_data, 0.8)

	print('\n\n')
	print('====================================')
	print('====           BLACK            ====')
	print('====================================')

	local epochs = {2,3,4,5,6,10}
	for i,n in ipairs(epochs) do
		print('Video corpus data: '..(n..' epochs'))
		trainAndTest(color_black_train, color_black_test, n)
		print('Video corpus data: '..(n..' epochs with ADAGRAD'))
		trainAndTest(color_black_train, color_black_test, n, true)
	end
end

local function testBlueClassifiers()
	local color_blue_data = torch.load('datasets/imagenet/attributes_datasets/blue.t7')

	local color_blue_train, color_blue_test = getTrainTestSplit(color_blue_data, 0.8)

	print('\n\n')
	print('====================================')
	print('====            BLUE            ====')
	print('====================================')

	print('Imagenet data: 2 epochs')
	trainAndTest(color_blue_train, color_blue_test, 2)
	print('Imagenet data: 3 epochs')
	trainAndTest(color_blue_train, color_blue_test, 3)
	print('Imagenet data: 4 epochs')
	trainAndTest(color_blue_train, color_blue_test, 4)
	print('Imagenet data: 5 epochs')
	trainAndTest(color_blue_train, color_blue_test, 5)
	print('Imagenet data: 6 epochs')
	trainAndTest(color_blue_train, color_blue_test, 6)
	print('Imagenet data: 10 epochs')
	trainAndTest(color_blue_train, color_blue_test, 10)
end

local function testRedClassifiers()
	local color_red_data = torch.load('datasets/imagenet/attributes_datasets/red.t7')

	local color_red_train, color_red_test = getTrainTestSplit(color_red_data, 0.8)

	print('\n\n')
	print('====================================')
	print('====            RED             ====')
	print('====================================')

	local epochs = {2,3,4,5,6,10}
	for i,n in ipairs(epochs) do
		print('Imagenet data: '..(n..' epochs'))
		trainAndTest(color_red_train, color_red_test, n)
		print('Imagenet data: '..(n..' epochs with ADAGRAD'))
		trainAndTest(color_red_train, color_red_test, n, true)
	end
end

local function testGrayClassifiers()
	local color_gray_data = torch.load('datasets/imagenet/attributes_datasets/gray.t7')

	local color_gray_train, color_gray_test = getTrainTestSplit(color_gray_data, 0.8)

	print('\n\n')
	print('====================================')
	print('====            GRAY            ====')
	print('====================================')

	print('Imagenet data: 2 epochs')
	trainAndTest(color_gray_train, color_gray_test, 2)
	print('Imagenet data: 3 epochs')
	trainAndTest(color_gray_train, color_gray_test, 3)
	print('Imagenet data: 4 epochs')
	trainAndTest(color_gray_train, color_gray_test, 4)
	print('Imagenet data: 5 epochs')
	trainAndTest(color_gray_train, color_gray_test, 5)
	print('Imagenet data: 6 epochs')
	trainAndTest(color_gray_train, color_gray_test, 6)
	print('Imagenet data: 10 epochs')
	trainAndTest(color_gray_train, color_gray_test, 10)
end

-- testGrayClassifiers()
-- testRedClassifiers()
-- testBlueClassifiers()
-- testBlackClassifiers()
testTrashbinClassifiers()
testBackpackClassifiers()
testChairClassifiers()
testPersonClassifiers()



