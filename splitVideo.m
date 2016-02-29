function splitVideo( vid_path, outpath, desired_side )

    originalVideo = VideoReader(vid_path);

    outputVideo = VideoWriter(outpath);
    open(outputVideo);
    
    while hasFrame(originalVideo)
        f = readFrame(originalVideo);
        frame_width = size(f, 2);

        if desired_side == 'L'
            half_f = f(:, 1:round(frame_width/2), :);
        elseif desired_side == 'R'
            half_f = f(:, round(frame_width/2):frame_width, :);
        else
            display('INVALID SIDE')
            return
        end
        writeVideo(outputVideo, half_f);
    end

    close(outputVideo);
end

