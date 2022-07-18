import os
import cv2
import skvideo
import argparse
import tqdm

class ExtractFrames(object):

    """Perform Center Crop on an given Image. Keeps the height to 256h fixed. Aspect ratio
    remains 1:1 to original frame.
    """
    def __init__(self, source_dir, output_dir, crop, sampling):

        if os.path.exists(source_dir):
            self.source_dir = source_dir
            self.video = cv2.VideoCapture(self.source_dir)
            self.fps = self.video.get(cv2.CAP_PROP_FPS)
            self.frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            raise FileExistsError(f"Video path: {source_dir} does not exist")
        self.crop = crop
        self.sampling = sampling
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        self.output_dir = output_dir


    # frame_id = 1 at the start of each label.
    # continuous ids inside each label.
    def extract_frames(self, idx, vid_dict, height=256):

        if idx == 0:
            frame_id = 0
        else:
            frame_id = idx*int(cv2.VideoCapture(vid_dict[idx-1]).get(cv2.CAP_PROP_FRAME_COUNT))

        success, frame = self.video.read()

        while success:
            if self.sampling == 'dense':
                frame_id+=1
            frame_name = os.path.join(self.output_dir, "{:05d}{}".format(frame_id, '.jpg'))
            resized_frame = cv2.resize(frame, (int((frame.shape[0]*frame.shape[1])/height), height))
            if self.crop == 'center':
                center = resized_frame.shape/2
                x = center[1] - resized_frame.shape[1]/2
                y = center[0] - height/2
                cropped_frame = resized_frame[y:y+height, x:x+resized_frame.shape[1]]
                cv2.imwrite(frame_name, cropped_frame)
            elif self.crop == 'normal':
                cv2.imwrite(frame_name, resized_frame)
            success, frame = self.video.read()

def get_frames(args):

    vid_dict = {}
    for label in tqdm.tqdm(os.listdir(args.source_dir)):
        label_root = os.path.join(args.source_dir, label)
        output_dir = os.path.join(args.output_dir, label)
        #items = len(next(os.walk(label_root))[2])
        for idx, video in enumerate(os.listdir(label_root)):
            source_dir = os.path.join(label_root, video)
            vid_dict[idx] = source_dir
            engine = ExtractFrames(source_dir=source_dir, output_dir=output_dir,
                             crop=args.crop, sampling=args.sampling)
            engine.extract_frames(idx=idx, vid_dict=vid_dict)

def main():

    parser = argparse.ArgumentParser(description='Extract Frames from videos')

    parser.add_argument('--crop',help='Image Crop', default='normal', choices = ['center', 'normal', 'corner'])
    parser.add_argument('--sampling', type = str, default='dense', choices = ['dense', 'uniform'])
    parser.add_argument('--split', default='train', choices = ['train', 'test', 'val'])

    args = parser.parse_args()

    args.source_dir = "/home/SarosijBose/HAR/KDHAR/soccer/" + args.split + '/'
    args.output_dir = "/home/SarosijBose/HAR/KDHAR/soccer/images/"+ args.split + '/'

    get_frames(args)

if __name__ == '__main__':
    main()