require 'torch'

local matio = require 'matio'
matio.use_lua_strings = true

dofile('/local/nrakover/meng/extract-features-and-opticalflow-from-detections.lua')

local function dirLookup(dir)
	local p = io.popen('find "'..dir..'" -type d')  --Open directory look for directories, save data in p. By giving '-type d' as parameter, it returns all directories.     
	local dir_names = {}
	for potential_dir in p:lines() do                      --Loop through all dirs
		if string.find(potential_dir, '/MVI_') ~= nil then
			table.insert(dir_names, potential_dir)
		end
	end
	return dir_names
end

function extractFeaturesAndOptflowFromAllVideos(source_dir, videos_to_process, compute_opticalflow)
	if compute_opticalflow == nil then
		compute_opticalflow = true
	end

	local all_vid_dirs = {}
	if videos_to_process == nil then
		all_vid_dirs = dirLookup(source_dir)
	else
		for i,vid in ipairs(videos_to_process) do
			all_vid_dirs[i] = source_dir..'/'..vid
		end
	end

	for i = 1, #all_vid_dirs do
		local vid_dir = all_vid_dirs[i]
		local vid_path = vid_dir..'/video.avi'
		local detections_path = vid_dir..'/detections.mat'

		local detections = matio.load(detections_path, 'detections_by_frame')
		local features, opticalflow = extractFeaturesAndOpticalFlow(detections, vid_path, compute_opticalflow)

		-- local old_features = torch.load(vid_dir..'/features.t7')
		-- for f = 1, #features do
		-- 	print('# new features: '..#features[f])
		-- 	for j = 1, #features[f] do
		-- 		old_features[f][#old_features[f] + 1] = features[f][j]:clone()
		-- 	end
		-- 	print('new total: '..#old_features[f])
		-- end

		torch.save(vid_dir..'/features.t7', features)
		if compute_opticalflow then
			torch.save(vid_dir..'/opticalflow.t7', opticalflow)
		end

		-- Display progress
		print('\t\t==> '..(100*i/#all_vid_dirs))
	end
end

local approach_vids = {
	'MVI_0822',
	'MVI_0823',
	'MVI_0824',
	'MVI_0825',
	'MVI_0826',
	'MVI_0827',
	'MVI_0835',
	'MVI_0836',
	'MVI_0837',
	'MVI_0848',
	'MVI_0850',
	'MVI_0855',
	'MVI_0856',
	'MVI_0857',
	'MVI_0858',
	'MVI_0859',
	'MVI_0868',
	'MVI_0869',
	'MVI_0870',
	'MVI_0882',
	'MVI_0884',
	'MVI_0885',
	'MVI_0886',
	'MVI_0887',
	'MVI_0888',
	'MVI_0889',
	'MVI_0890',
	'MVI_0891',
	'MVI_0898',
	'MVI_0900',
	'MVI_0913',
	'MVI_0915',
	'MVI_0107',
	'MVI_0108',
	'MVI_0109',
	'MVI_0110',
	'MVI_0111'
}

local pickup_vids = {
	'MVI_0838',
	'MVI_0839',
	'MVI_0840',
	'MVI_0841',
	'MVI_0843',
	'MVI_0844',
	'MVI_0845',
	'MVI_0846',
	'MVI_0847',
	'MVI_0871',
	'MVI_0872',
	'MVI_0874',
	'MVI_0875',
	'MVI_0877',
	'MVI_0878',
	'MVI_0879',
	'MVI_0880',
	'MVI_0881',
	'MVI_0901',
	'MVI_0902',
	'MVI_0903',
	'MVI_0904',
	'MVI_0907',
	'MVI_0908',
	'MVI_0909',
	'MVI_0910',
	'MVI_0911',
	'MVI_0912',
	'MVI_0854',
	'MVI_0820',
	'MVI_0821',
	'MVI_0883'
}

local put_down_vids = {
	'MVI_0828',
	'MVI_0829',
	'MVI_0830',
	'MVI_0831',
	'MVI_0832',
	'MVI_0833',
	'MVI_0834',
	'MVI_0842',
	'MVI_0844',
	'MVI_0860',
	'MVI_0861',
	'MVI_0862',
	'MVI_0863',
	'MVI_0864',
	'MVI_0865',
	'MVI_0866',
	'MVI_0867',
	'MVI_0876',
	'MVI_0878',
	'MVI_0892',
	'MVI_0893',
	'MVI_0894',
	'MVI_0895',
	'MVI_0896',
	'MVI_0897',
	'MVI_0905',
	'MVI_0906',
	'MVI_0907',
	'MVI_0112',
	'MVI_0113',
	'MVI_0114',
	'MVI_0115',
	'MVI_0116'
}

local pickup_single_track_vids = {
	'MVI_0838a',
	'MVI_0839a',
	'MVI_0840a',
	'MVI_0840b',
	'MVI_0841a',
	'MVI_0843a',
	'MVI_0844a',
	'MVI_0845a',
	'MVI_0846a',
	'MVI_0847a',
	'MVI_0847b',
	'MVI_0871a',
	'MVI_0872a',
	'MVI_0874a',
	'MVI_0874b',
	'MVI_0875a',
	'MVI_0877a',
	'MVI_0878a',
	'MVI_0879a',
	'MVI_0881a',
	'MVI_0881b',
	'MVI_0901a',
	'MVI_0902a',
	'MVI_0903a',
	'MVI_0903b',
	'MVI_0904a',
	'MVI_0907a',
	'MVI_0908a',
	'MVI_0909a',
	'MVI_0910a',
	'MVI_0910b',
	'MVI_0911a',
	'MVI_0911b',
	'MVI_0912a',
	'MVI_0912b',
	'MVI_0854a',
	'MVI_0820a',
	'MVI_0821a',
	'MVI_0822a',
	'MVI_0823a',
	'MVI_0825a',
	'MVI_0825b',
	'MVI_0826a',
	'MVI_0827a',
	'MVI_0837a',
	'MVI_0837b',
	'MVI_0837c',
	'MVI_0884a',
	'MVI_0838b',
	'MVI_0838c',
	'MVI_0838d',
	'MVI_0838e',
	'MVI_0839b',
	'MVI_0839c',
	'MVI_0840c',
	'MVI_0841b',
	'MVI_0843b',
	'MVI_0844b',
	'MVI_0845b',
	'MVI_0846b',
	'MVI_0846c',
	'MVI_0847c',
	'MVI_0871b',
	'MVI_0872b',
	'MVI_0872c',
	'MVI_0874c',
	'MVI_0874d',
	'MVI_0874e',
	'MVI_0875b',
	'MVI_0875c',
	'MVI_0875d',
	'MVI_0877b',
	'MVI_0877c',
	'MVI_0877d',
	'MVI_0878b',
	'MVI_0879b',
	'MVI_0881c',
	'MVI_0901b',
	'MVI_0901c',
	'MVI_0902b',
	'MVI_0903c',
	'MVI_0903d',
	'MVI_0903e',
	'MVI_0904b',
	'MVI_0907b',
	'MVI_0908b',
	'MVI_0909b',
	'MVI_0910c',
	'MVI_0910d',
	'MVI_0911c',
	'MVI_0912c',
	'MVI_0840T',
	'MVI_0843T',
	'MVI_0844T',
	'MVI_0847T',
	'MVI_0874T',
	'MVI_0877T',
	'MVI_0878T',
	'MVI_0881T',
	'MVI_0903T',
	'MVI_0910T',
	'MVI_0911T',
	'MVI_0912T',
	'MVI_0101a',
	'MVI_0102a',
	'MVI_0103a',
	'MVI_0104a',
	'MVI_0105a',
	'MVI_0106a',
	'MVI_0839d',
	'MVI_0841c',
	'MVI_0854b',
	'MVI_0821b',
	'MVI_0822b',
	'MVI_0872d',
	'MVI_0875e',
	'MVI_0875f',
	'MVI_0879c',
	'MVI_0883a',
	'MVI_0901d',
	'MVI_0902c',
	'MVI_0904c',
	'MVI_0908c',
	'MVI_0909c'
}

local leave_vids = {
	'MVI_0836',
	'MVI_0849',
	'MVI_0850',
	'MVI_0869',
	'MVI_0883',
	'MVI_0884',
	'MVI_0899',
	'MVI_0914',
	'MVI_0915'
}

-- local leave_vids = {
-- 	'MVI_0849',
-- 	'MVI_0899',
-- 	'MVI_0914'
-- }

local half_frame_pickup_vids = {
	'MVI_0101a',
	'MVI_0102a',
	'MVI_0103a',
	'MVI_0104a',
	'MVI_0105a',
	'MVI_0106a'
}

local book_vids = {
	'MVI_0201',
	'MVI_0202',
	'MVI_0203',
	'MVI_0204',
	'MVI_0205',
	'MVI_0206',
	'MVI_0207',
	'MVI_0208',
	'MVI_0209',
	'MVI_0210',
	'MVI_0211',
	'MVI_0212',
	'MVI_0213',
	'MVI_0214',
	'MVI_0215',
	'MVI_0216',
	'MVI_0217',
	'MVI_0218',
	'MVI_0219',
	'MVI_0220',
	'MVI_0221',
	'MVI_0222',
	'MVI_0223',
	'MVI_0224',
	'MVI_0225',
	'MVI_0226',
	'MVI_0227',
	'MVI_0228',
	'MVI_0229',
	'MVI_0230',
	'MVI_0231',
	'MVI_0232',
	'MVI_0233',
	'MVI_0234',
	'MVI_0235',
	'MVI_0236',
	'MVI_0237',
	'MVI_0238',
	'MVI_0239',
	'MVI_0240',
	'MVI_0241',
	'MVI_0242',
	'MVI_0243'
}

local lamp_vids = {
	'MVI_0244',
	'MVI_0245',
	'MVI_0246',
	'MVI_0247',
	'MVI_0248',
	'MVI_0249',
	'MVI_0250',
	'MVI_0251',
	'MVI_0252',
	'MVI_0253',
	'MVI_0254',
	'MVI_0255',
	'MVI_0256',
	'MVI_0257',
	'MVI_0258',
	'MVI_0259',
	'MVI_0260',
	'MVI_0261',
	'MVI_0262',
	'MVI_0263',
	'MVI_0264',
	'MVI_0265',
	'MVI_0266',
	'MVI_0267',
	'MVI_0268',
	'MVI_0269',
	'MVI_0270',
	'MVI_0271',
	'MVI_0272',
	'MVI_0273',
	'MVI_0274',
	'MVI_0275',
	'MVI_0276',
	'MVI_0277',
	'MVI_0278',
	'MVI_0279',
	'MVI_0280',
	'MVI_0281',
	'MVI_0282',
	'MVI_0283',
	'MVI_0284',
	'MVI_0285',
	'MVI_0286',
	'MVI_0287',
	'MVI_0288',
	'MVI_0289',
	'MVI_0290',
	'MVI_0291'
}

local walk_skip_vids = {
	'MVI_0301',
	'MVI_0302',
	'MVI_0303',
	'MVI_0304',
	'MVI_0305',
	'MVI_0306',
	'MVI_0307',
	'MVI_0308',
	'MVI_0309',
	'MVI_0310',
	'MVI_0311',
	'MVI_0312',
	'MVI_0313',
	'MVI_0314',
	'MVI_0315',
	'MVI_0316',
	'MVI_0317',
	'MVI_0318',
	'MVI_0319',
	'MVI_0320',
	'MVI_0321',
	'MVI_0322',
	'MVI_0323',
	'MVI_0324',
	'MVI_0325',
	'MVI_0326',
	'MVI_0327',
	'MVI_0328',
	'MVI_0329',
	'MVI_0330',
	'MVI_0331',
	'MVI_0332',
	'MVI_0333',
	'MVI_0334',
	'MVI_0335',
	'MVI_0336',
	'MVI_0337',
	'MVI_0338',
	'MVI_0339',
	'MVI_0340'
}

local fix_list = {
	'MVI_0112',
	'MVI_0113',
	'MVI_0114',
	'MVI_0115',
	'MVI_0116'
}

extractFeaturesAndOptflowFromAllVideos('single_track_videos', put_down_vids, true)
-- extractFeaturesAndOptflowFromAllVideos('videos_272x192_500_proposals', approach_vids)





