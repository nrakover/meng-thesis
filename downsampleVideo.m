function downsampleVideo( vidPath, newPath, frameDownsampleRate, sizeDownsampleRate )

vid = VideoReader(vidPath);
skipNum = round(1/frameDownsampleRate);

outputVideo = VideoWriter(newPath);
open(outputVideo);

fCount = 1;
while hasFrame(vid)
   f = readFrame(vid);
   if mod(fCount, skipNum) == 1
       writeVideo(outputVideo,imresize(f,sizeDownsampleRate));
   end
   fCount = fCount + 1;
end

close(outputVideo);
end