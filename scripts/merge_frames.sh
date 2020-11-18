set -ex

cd $1

# composite frame to video
/usr/bin/ffmpeg -hide_banner -y -loglevel warning \
    -thread_queue_size 8192 -i render/%05d.png \
    -thread_queue_size 8192 -i crop/%05d.png \
    -thread_queue_size 8192 -i overlay/%05d.png \
    -i audio/audio.aac -filter_complex hstack=inputs=3 \
    -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p output.mp4

cd -
