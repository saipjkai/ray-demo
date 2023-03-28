# internal
from argparse import ArgumentParser
import os

# external
import numpy as np
import cv2
from ultralytics import YOLO

# user-defined
from utils.data_utils import vid2frames, show_frames
from utils.yolo_utils import load_classes, draw_predictions, save_results


def main(args):
    # data
    if args.video is not None:
        video_path = args.video
        imgs = vid2frames(video_path)
    else:
        img_path = args.img
        imgs = [cv2.imread(img_path)]

    # load model
    weights_path = args.weights
    model = YOLO(weights_path)  # load a YOLO arch with pretrained weights (recommended for training)
    
    # get class labels
    classes_path = args.classes
    classes = load_classes(classes_path)
    
    # get predictions
    predictions = []
    for img in imgs:
        prediction = model.predict(img)
        predictions.extend(prediction)

    # draw predictions
    results = draw_predictions(imgs=imgs, predictions=predictions, classes=classes)
    
    # show predictions
    show_frames(imgs=results)

    # save output
    results_dir = args.output_dir
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    save_results(results, results_dir)


def get_args():
    ap = ArgumentParser()
    ap.add_argument("--img", default=os.path.join(os.getcwd(), "data", "images", "bus.jpg"), help="Path to image")
    ap.add_argument("--video", default=None, help="Path to video")
    ap.add_argument("--weights", default=os.path.join(os.getcwd(), "yolo", "weights", "yolov8n.pt"), help="Path to YOLO weights")
    ap.add_argument("--classes", default=os.path.join(os.getcwd(), "yolo", "labels", "coco_labels.txt"), help="Path to YOLO labels")
    ap.add_argument("--output_dir", default=os.path.join(os.getcwd(), "yolo", "results"), help="Directory to store results")

    args = ap.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()

    main(args)

