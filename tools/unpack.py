# Unpack k400 .tar.gz files into mp4 files

import os
import tarfile
import tqdm

#hardcode paths
train_root = r'/home/SarosijBose/HAR/KDHAR/kinetics400/k400/train'
test_root = r'/home/SarosijBose/HAR/KDHAR/kinetics400/k400/test'
val_root = r'/home/SarosijBose/HAR/KDHAR/kinetics400/k400/val'

train_path = r'/home/SarosijBose/HAR/KDHAR/kinetics400/k400/train_v'
test_path = r'/home/SarosijBose/HAR/KDHAR/kinetics400/k400/test_v'
val_path = r'/home/SarosijBose/HAR/KDHAR/kinetics400/k400/val_v'

def main():

    for filename in tqdm.tqdm(os.listdir(train_root)):
        item  = os.path.join(train_root, filename)
        if '.gz' in filename:
            videos = tarfile.open(item)
            if not os.path.exists(train_path):
                os.mkdir(train_path)
            videos.extractall(train_path)
            videos.close()

    for filename in tqdm.tqdm(os.listdir(test_root)):
        item  = os.path.join(test_root, filename)
        if '.gz' in filename:
            videos = tarfile.open(item)
            if not os.path.exists(test_path):
                os.mkdir(test_path)
            videos.extractall(test_path)
            videos.close()

    for filename in tqdm.tqdm(os.listdir(val_root)):
        item  = os.path.join(val_root, filename)
        if '.gz' in filename:
            videos = tarfile.open(item)
            if not os.path.exists(val_path):
                os.mkdir(val_path)
            videos.extractall(val_path)
            videos.close()

    print('Kinetics 400 extraction complete!')

main()