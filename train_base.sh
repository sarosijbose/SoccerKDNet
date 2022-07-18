CUDA_VISIBLE_DEVICES=1 python3 train.py --datadir /home/SarosijBose/HAR/KDHAR/kinetics400/k400 \
--dataset kinetics400 --frames_per_group 1 --groups 8 \
--logdir snapshots/ --lr 0.01 --backbone_net resnet -d 50 -b 16 -j 64 --temporal_module_name TAM