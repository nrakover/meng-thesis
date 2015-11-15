function visualizeTrack( track_filename, video_filename )

	v = VideoReader(video_filename);

	track = load(track_filename);
	n_frames = int16(track.nFrames(1,1));

	for i = 1:n_frames
	    frame = readFrame(v);
	    bbox = track.(strcat('t',num2str(i)));
	    
	    x = bbox(1);
	    y = bbox(2);
	    w = bbox(3) - x;
	    h = bbox(4) - y;
	    
	    figure;
	    imshow(frame);
	    hold on
	    rectangle('Position', [x y w h], 'EdgeColor', 'r');
	    hold off
	end

end

