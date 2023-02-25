import os
import cv2
import tqdm
from copy import deepcopy
import numpy as np
import ipdb

def compile_video(out_dir, out_dir_2):
    """
    Writes frames to an mp4 video file
    param file_path: Path to output video, must end with .mp4
    """

    # create container for frames which will hold all images
    frames = []

    # hard set FPS to 30
    FPS = 30

    # output video path
    out_video_path = os.path.join(out_dir, 'robotdata2.mp4')

    # create dummny list to store each frame
    if out_dir_2 is not None:
        frames = list(range(1, len(os.listdir(out_dir)) + len(os.listdir(out_dir_2)) + 2))
        print("frame length is", len(frames))

        # compile the frames
        for img_name in os.listdir(out_dir):
            if img_name.endswith(".png"):
                # frame_no = int(img_name.split(".")[0].split("_")[1])
                # frames[frame_no]= (cv2.imread(os.path.join(out_dir, img_name)))
                img_name_orig = img_name
                img_name2 = img_name_orig.split(".png")[0]
                img_name2 = img_name2.lstrip("0")
                # print(img_name2)
                frame_no = int(img_name2)
                frames[frame_no]= cv2.imread(os.path.join(out_dir, img_name_orig))

        # compile the frames
        for img_name in os.listdir(out_dir_2):
            if img_name.endswith(".png"):
                # frame_no = int(img_name.split(".")[0].split("_")[1])
                # frames[frame_no]= (cv2.imread(os.path.join(out_dir, img_name)))
                img_name_orig = img_name
                img_name2 = img_name_orig.split(".png")[0]
                img_name2 = img_name2.lstrip("0")
                # print(img_name2)
                frame_no = int(img_name2)
                frames[frame_no]= cv2.imread(os.path.join(out_dir, img_name_orig))

    else:
        frames = list(range(1, len(os.listdir(out_dir)) + 2))
        print("frame length is", len(frames))

        # compile the frames
        for img_name in os.listdir(out_dir):
            if img_name.endswith(".png"):
                # frame_no = int(img_name.split(".")[0].split("_")[1])
                # frames[frame_no]= (cv2.imread(os.path.join(out_dir, img_name)))
                img_name_orig = img_name
                img_name2 = img_name_orig.split(".png")[0]
                img_name2 = img_name2.lstrip("0")
                # print(img_name2)
                frame_no = int(img_name2)
                frames[frame_no]= cv2.imread(os.path.join(out_dir, img_name_orig))

    print("finished reading images")
    frames = frames[1:]
    h, w, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(out_video_path, fourcc , FPS, (w, h))

    print("composing AR frames into video")
    for f in frames:
        if isinstance(f, np.ndarray):
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            writer.write(f)

    writer.release()

if __name__ == '__main__':
    compile_video('/home/sush/CMU/SLAM-and-Robot-Autonomy/SLAM/Particle Filters/16833_HW1_ParticleFilter_Starter_Code/code/results_3841_log2', None)