CUDA_VISIBLE_DEVICES=1 python test.py --groups 32 -e --frames_per_group 2 --without_t_stride --logdir logs/ --dataset kinetics400 \
--num_crops 3 --num_clips 10 --input_size 256 --disable_scaleup -b 6 --gpu 1 --dense_sampling \
--datadir /home/SarosijBose/HAR/KDHAR/kinetics400/k400 \
--backbone_net resnet -d 50 --temporal_module_name TAM \
--pretrained ./state_dicts/K400-TAM-ResNet-50-f32.pth.tar 