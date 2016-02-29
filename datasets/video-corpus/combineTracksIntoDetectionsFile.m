function combineTracksIntoDetectionsFile( detections_struct_path, track_paths, outfile )

    D = load(detections_struct_path);
    num_frames = size(D.detections_by_frame.detections, 1);

    num_tracks = size(track_paths, 2);

    detections = zeros(num_frames, num_tracks, 4);
    for tIndx = 1:num_tracks
        track = load(track_paths{tIndx});
        for fIndx = 1:num_frames
            detections(fIndx, tIndx, :) = track.(strcat('t',num2str(fIndx)));
        end
    end

    detections_by_frame = struct('detections', detections, 'height', D.detections_by_frame.height, 'width', D.detections_by_frame.width, 'fps', D.detections_by_frame.fps, 'length', D.detections_by_frame.length);
    save(outfile, 'detections_by_frame')

end

