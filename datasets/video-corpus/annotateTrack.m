function annotateTrack( dir_path, track_name )

    vid_path = [dir_path  '/video.avi'];
    vid = VideoReader(vid_path);
    
    track = struct();
    f = 0;
    while hasFrame(vid)
        f = f + 1;
        frame = readFrame(vid);
        imshow(frame)
        rect = getrect;
        
        bbox = [];
        bbox(1) = rect(1);
        bbox(2) = rect(2);
        bbox(3) = rect(1) + rect(3) - 1;
        bbox(4) = rect(2) + rect(4) - 1;
        
        track.(strcat('t',num2str(f))) = bbox;
    end
    
    track.nFrames = f;
    
    track_path = [dir_path '/' track_name '.mat'];
    save(track_path, '-struct', 'track');
end

