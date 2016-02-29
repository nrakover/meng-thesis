function retrofitAddPersonDetections( video_path, detections_path )

	peopleDetector1 = vision.PeopleDetector;
	peopleDetector1.ClassificationThreshold = 0.01;
	peopleDetector1.ClassificationModel = 'UprightPeople_96x48';
	peopleDetector2 = vision.PeopleDetector;
	peopleDetector2.ClassificationThreshold = 0.01;
	peopleDetector2.ClassificationModel = 'UprightPeople_128x64';

	vid = VideoReader(video_path);
	SAVED_DATA = load(detections_path);
	detections_by_frame = SAVED_DATA.detections_by_frame;
	old_detections = detections_by_frame.detections;

	num_frames = size(old_detections, 1);
	person_detections_by_frame = cell(num_frames,1);

	max_num_detections = 0;
	fIndx = 1;
	while hasFrame(vid)
		frame = readFrame(vid);
		
		[bboxes1,scores1] = step(peopleDetector1,frame);
		[bboxes2,scores2] = step(peopleDetector2,frame);

		bboxes = convertBBoxesToMinMaxMinMax([bboxes1 ; bboxes2]);
		person_detections_by_frame{fIndx} = bboxes;
		if size(bboxes, 1) > max_num_detections
			max_num_detections = size(bboxes, 1);
		end

		fIndx = fIndx + 1;
	end

	num_old_detections = size(old_detections, 2);
	new_detections = zeros( num_frames, num_old_detections + max_num_detections, 4 );
	person_detector_indices = zeros(num_frames, 2);
	for fIndx = 1:num_frames
		new_detections(fIndx, 1:num_old_detections, :) = old_detections(fIndx, :, :);

		num_person_detections = size(person_detections_by_frame{fIndx}, 1);
		new_detections(fIndx, num_old_detections+1:num_old_detections+num_person_detections, :) = person_detections_by_frame{fIndx}(:,:);

		person_detector_indices(fIndx,1) = num_old_detections+1;
		person_detector_indices(fIndx,2) = num_old_detections+num_person_detections;
	end

	detections_by_frame.detections = new_detections;
	detections_by_frame.person_detector_indices = person_detector_indices;
	save(detections_path, 'detections_by_frame')
end


function new_bboxes = convertBBoxesToMinMaxMinMax( bboxes )

	new_bboxes = zeros(size(bboxes));
	for i = 1:size(bboxes, 1)
		new_bboxes(i, 1) = bboxes(i, 1);
		new_bboxes(i, 2) = bboxes(i, 2);
		new_bboxes(i, 3) = bboxes(i, 1) + bboxes(i, 3) - 1;
		new_bboxes(i, 4) = bboxes(i, 2) + bboxes(i, 4) - 1;
	end

end

