function detections_by_frame = videoToDetections( infile, outfile, numProposals )
	% addpath('objectness/object-proposals/');
	% initialize;

	v = VideoReader(infile);
	numFrames = floor(v.Duration * v.FrameRate) - 1;

	detections = zeros(numFrames, numProposals, 4);

	for t = 1:numFrames
		if hasFrame(v)
			I = readFrame(v);
			frame_proposals = ObjectProposals('randomPrim', I, numProposals);

			detections(t,:,:) = frame_proposals.boxes;
		end
		display(100*t/numFrames)
	end

	detections_by_frame = struct('detections', detections, 'height', v.Height, 'width', v.Width, 'fps', v.FrameRate, 'length', v.Duration);
	save(outfile, 'detections_by_frame')
end