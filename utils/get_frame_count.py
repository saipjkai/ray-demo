from argparse import ArgumentParser

import os
import cv2

from data_utils import vid2frames


def main(args):
    video_path = args.input_video
    frames = vid2frames(video_path)
    print("{} contains : {} frames".format(video_path, len(frames)))


def get_args():
    ap = ArgumentParser()
    ap.add_argument("--input_video", default=os.path.abspath(os.path.join(os.getcwd(), "data/videos/demo.mp4")), help="Input video path to get number of frames in it.")

    args = ap.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    main(args)
