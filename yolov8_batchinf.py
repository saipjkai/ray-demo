# internal
from argparse import ArgumentParser
import os

# external
from ultralytics import YOLO

# user-defined
from utils.data_utils import vid2frames, batch_generator, show_frames 
from utils.yolo_utils import load_classes, draw_predictions, save_results


def main(args):
    # data
    video_path = args.video
    frames = vid2frames(video_path)

    # batches 
    batches = batch_generator(frames, batch_size=8)

    # load model
    weights_path = args.weights
    model = YOLO(weights_path)

    # get class labels
    classes_path = args.classes
    classes = load_classes(classes_path)

    # get predictions
    predictions = []
    for batch in batches:
        outs = model(batch)
        predictions.append(outs)

    # draw predictions
    results = []
    for batch, prediction in zip(batches, predictions):
        outs = draw_predictions(batch, prediction, classes)
        results.extend(outs)

    # show output
    show_frames(imgs=results)

    # save output
    results_dir = args.output_dir
    save_results(results, results_dir)


def get_args():
    ap = ArgumentParser()
    ap.add_argument("--video", default=os.path.join(os.getcwd(), "data", "videos", "demo.mp4"), help="Path to video")
    ap.add_argument("--weights", default=os.path.join(os.getcwd(), "yolo", "weights", "yolov8n.pt"), help="Path to YOLO weights")
    ap.add_argument("--classes", default=os.path.join(os.getcwd(), "yolo", "labels", "coco_labels.txt"), help="Path to YOLO labels")
    ap.add_argument("--output_dir", default=os.path.join(os.getcwd(), "yolo", "results"), help="Directory to store results")

    args = ap.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    main(args)
