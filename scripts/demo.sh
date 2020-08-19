set -ex

# set data path
video_dir="data/video"
audio_dir="data/test_audio"

# # create directory
# mkdir -p $video_dir/full
# mkdir -p $video_dir/crop
# mkdir -p $video_dir/audio

# set video clip duration
start_time="00:00:00"
end_time="240"


# extract video frames and audio
# /usr/bin/ffmpeg -hide_banner -y -i $video_dir/*.mp4 -ss $start_time -t $end_time -r 25 $video_dir/full/%05d.png
# /usr/bin/ffmpeg -hide_banner -y -i $video_dir/*.mp4 -ss $start_time -t $end_time $video_dir/audio/audio.aac


# extract high-level feature from audio
# mkdir -p $video_dir/feature
# python vendor/ATVGnet/code/test.py -i $video_dir/


# extract high-level feature from test audio
# mkdir -p $audio_dir/feature
# python vendor/ATVGnet/code/test.py -i $audio_dir/


# # crop and resize video frames
# python utils/crop_portrait.py \
#    --data_dir $video_dir \
#    --crop_level 2.0 \
#    --vertical_adjust 0.2


# # # 3D face reconstruction
# python train.py \
#     --data_dir $video_dir \
#     --num_epoch 20 \
#     --serial_batches False \
#     --display_freq 400 \
#     --print_freq 400 \
#     --batch_size 5


# build neural face renderer data pair
# python utils/build_nfr_dataset.py --data_dir $video_dir


# # # create reconstruction debug video
# /usr/bin/ffmpeg -hide_banner -y -loglevel warning \
#     -thread_queue_size 8192 -i $video_dir/render/%05d.png \
#     -thread_queue_size 8192 -i $video_dir/crop/%05d.png \
#     -thread_queue_size 8192 -i $video_dir/overlay/%05d.png \
#     -i $video_dir/audio/audio.aac \
#     -filter_complex hstack=inputs=3 -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p $video_dir/debug.mp4


# # # train audio2expression network
# python train_exp.py \
#     --dataset_mode audio_expression \
#     --num_epoch 10 \
#     --serial_batches False \
#     --display_freq 800 \
#     --print_freq 800 \
#     --batch_size 5 \
#     --lr 1e-3 \
#     --lambda_delta 1.0 \
#     --data_dir $video_dir \
#     --net_dir $video_dir



# # train neural face renderer
# python vendor/neural-face-renderer/train.py 
#     --dataroot $video_dir/nfr/AB --name nfr --model nfr --checkpoints_dir $video_dir/checkpoints \
#     --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode temporal --norm batch --pool_size 0 --use_refine \
#     --input_nc 21 --Nw 7 --batch_size 16 --preprocess none --num_threads 4 --n_epochs 250 \
#     --n_epochs_decay 0 --load_size 256


# predict expression parameter fron audio feature
# python test_exp.py --dataset_mode audio_expression \
#     --data_dir $audio_dir \
#     --net_dir $video_dir


# reenact face using predicted expression parameter
# python reenact.py --src_dir $audio_dir --tgt_dir $video_dir


# choose best epoch with lowest loss
# epoch=50

# neural rendering the reenact face sequence
# python vendor/neural-face-renderer/test.py --model test \
#     --netG unet_256 \
#     --direction BtoA \
#     --dataset_mode temporal_single \
#     --norm batch \
#     --input_nc 21 \
#     --Nw 7 \
#     --preprocess none \
#     --eval \
#     --use_refine \
#     --name nfr \
#     --checkpoints_dir $video_dir/checkpoints \
#     --dataroot $audio_dir/reenact \
#     --results_dir $audio_dir \
#     --epoch $epoch

# composite lower face back to original video
# python comp.py --src_dir $audio_dir --tgt_dir $video_dir


# create final result
# /usr/bin/ffmpeg -y -loglevel warning \
#     -thread_queue_size 8192 -i $audio_dir/audio/audio.aac \
#     -thread_queue_size 8192 -i $audio_dir/comp/%05d.png \
#     -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest $audio_dir/result.mp4
