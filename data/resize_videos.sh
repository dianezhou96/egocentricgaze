for d in */ ; do
    ffmpeg -i $d/world.mp4 -vf "format=yuv444p,scale=227:227" $d/world_resized.mp4
done
