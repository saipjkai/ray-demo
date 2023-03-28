# internal
from argparse import ArgumentParser
import os

# external
import ray
from ultralytics import YOLO

# user-defined
from utils.data_utils import vid2frames, batch_generator, show_frames
from utils.yolo_utils import load_classes, draw_predictions, save_results


@ray.remote
def predict(model, images):
    return model(images)


def main(args):
    # data
    video_path = args.video
    frames = vid2frames(video_path)

    # batches 
    batches = batch_generator(frames, batch_size=8)

    # model
    weights_path = args.weights
    model = YOLO(weights_path)
    model_ref = ray.put(model)

    # get class labels
    classes_path = args.classes
    classes = load_classes(classes_path)

    # get predictions
    prediction_refs = []
    predictions = []
    for batch in batches:
        outs = predict.remote(model_ref, batch)
        prediction_refs.append(outs)

    predictions = ray.get(prediction_refs)

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
    # args
    args = get_args()

    # ray initialization
    ray.init(address='auto') # ray start --head --port=6379

    print("--"*50)
    print('''This cluster consists of
            {} nodes in total
            {} CPU resources in total'''.format(len(ray.nodes()), ray.cluster_resources()['CPU'])
        )
    print("--"*50)

    main(args)
