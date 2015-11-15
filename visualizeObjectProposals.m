function visualizeObjectProposals( proposals_filename, video_filename )

    proposals_data = load(proposals_filename);
    proposals = proposals_data.detections_by_frame.detections;
    v = VideoReader(video_filename);

    n_frames = size(proposals, 1);

    for i = 1:n_frames
        figure;
        
        frame = readFrame(v);
        imshow(frame)
        hold on
        
        for j = 1:size(proposals,2)
            bbox = proposals(i,j,:);
            x = bbox(1);
            y = bbox(2);
            w = bbox(3) - x;
            h = bbox(4) - y;
            
            rectangle('Position', [x y w h], 'EdgeColor', 'r');
        end
        hold off
    end

end

