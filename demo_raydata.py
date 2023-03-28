from argparse import ArgumentParser
import os

import numpy as np
import cv2

import ray

def vid2frames(video_path):
    try:
        vc = cv2.VideoCapture(video_path)

        frames = []

        while vc.isOpened():
            success, frame = vc.read()
            if success:
                frames.append(frame)
            else:
                break
        vc.release()

        return frames
    
    except:
        print("Video not found or corrupted!")
        exit(0)


def main(args):
    # video path
    video_path = args.input_video

    # frames
    frames = vid2frames(video_path)

    # frames to numpy array
    frames_npy = np.array(frames)

    # numpy array to ray data
    ray_ds = ray.data.from_numpy(frames_npy)
    print(ray_ds.take(1))


def get_args():
    ap = ArgumentParser()
    ap.add_argument("--input_video", default=os.path.abspath(os.path.join(os.getcwd(), "data/videos/demo.mp4")), help="Input video path to get number of frames in it.")

    args = ap.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    main(args)