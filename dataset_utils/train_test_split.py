import os
import shutil

train_root = r'/home/SarosijBose/HAR/KDHAR/soccer/train'
test_root = r'/home/SarosijBose/HAR/KDHAR/soccer/test'
dir_count = [70, 92, 151, 135]

for label_id, label in enumerate(os.listdir(train_root)):
    class_root = os.path.join(train_root, label)
    for idx, video in enumerate(os.listdir(class_root)):
        if (idx+1) >= int(0.8 * dir_count[label_id]):
            source_dir = os.path.join(class_root, video)
            target_dir = os.path.join(test_root, label)
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
            shutil.move(source_dir, target_dir)
        else:
            pass