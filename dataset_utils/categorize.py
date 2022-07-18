import os
import shutil
from csv import reader
import tqdm

#hardcode paths
train_root = r'/home/SarosijBose/HAR/KDHAR/kinetics400/k400/train'
test_root = r'/home/SarosijBose/HAR/KDHAR/kinetics400/k400/test'
val_root = r'/home/SarosijBose/HAR/KDHAR/kinetics400/k400/val'

root = r'/home/SarosijBose/HAR/KDHAR/kinetics400/k400'

# Organize videos to class folders
def main():

    for video in tqdm.tqdm(os.listdir(train_root)):
        #__SzQTzL7_I_000002_000012
        id = video.split('_0')[0]
        train_labels = train_root + '.csv'
        with open(train_labels, 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            if header != None:
                for row in csv_reader:
                    label = row[0]
                    target_dir = os.path.join(train_root, label)
                    if row[1] == id:
                        if not os.path.exists(target_dir):
                            os.mkdir(target_dir)
                        shutil.move(os.path.join(train_root, video), target_dir)
        
    for video in tqdm.tqdm(os.listdir(test_root)):
        #__SzQTzL7_I_000002_000012
        id = video.split('_0')[0]
        test_labels = test_root + '.csv'
        with open(test_labels, 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            if header != None:
                for row in csv_reader:
                    label = row[0]
                    target_dir = os.path.join(test_root, label)
                    if row[1] == id:
                        if not os.path.exists(target_dir):
                            os.mkdir(target_dir)
                        shutil.move(os.path.join(test_root, video), target_dir)

    for video in tqdm.tqdm(os.listdir(val_root)):
        #__SzQTzL7_I_000002_000012
        id = video.split('_0')[0]
        val_labels = val_root + '.csv'
        with open(val_labels, 'r') as read_obj:
            csv_reader = reader(read_obj)
            header = next(csv_reader)
            if header != None:
                for row in csv_reader:
                    label = row[0]
                    target_dir = os.path.join(val_root, label)
                    if row[1] == id:
                        if not os.path.exists(target_dir):
                            os.mkdir(target_dir)
                        shutil.move(os.path.join(val_root, video), target_dir)

# Check for class anomalies
def find_anomaly():

    flag = False

    for train_item in tqdm.tqdm(os.listdir(train_root)):
        if '.mp4' not in train_item:
            for test_item in os.listdir(test_root):
                if '.mp4' not in test_item:
                    if train_item == test_item:
                        flag = True
                        break
            if flag:
                print(train_item)
                break
            flag  = False

    flag = False
    for train_item in tqdm.tqdm(os.listdir(train_root)):
        if '.mp4' not in train_item:
            for val_item in os.listdir(val_root):
                if '.mp4' not in val_item:
                    if train_item == val_item:
                        flag = True
                        break
            if flag:
                print(train_item)
                break
            flag  = False

def remove_extras():

    train_move_target = os.path.join(root, 'train_extras')
    if not os.path.exists(train_move_target):
        os.mkdir(train_move_target)
    for train_item in tqdm.tqdm(os.listdir(train_root)):
        if '.mp4' in train_item:
            shutil.move(os.path.join(train_root, train_item), train_move_target)

    test_move_target = os.path.join(root, 'test_extras')
    if not os.path.exists(test_move_target):
        os.mkdir(test_move_target)
    for test_item in tqdm.tqdm(os.listdir(test_root)):
        if '.mp4' in test_item:
            shutil.move(os.path.join(test_root, test_item), test_move_target)

    val_move_target = os.path.join(root, 'val_extras')
    if not os.path.exists(val_move_target):
        os.mkdir(val_move_target)
    for val_item in tqdm.tqdm(os.listdir(val_root)):
        if '.mp4' in val_item:
            shutil.move(os.path.join(val_root, val_item), val_move_target)
                  
#main()
#find_anomaly()
remove_extras()