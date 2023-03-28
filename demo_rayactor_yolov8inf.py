# internal
from argparse import ArgumentParser
import os
import time

# external
import numpy
import cv2

import torch
from ultralytics import YOLO

import ray

# user-defined
from utils.data_utils import vid2frames, batch_generator


def prediction_results_postprocessing(predictions,  outs):
    predictions.append(outs)
    return predictions


@ray.remote
class PredictActor:
    def __init__(self, model):
        self.model = model
    
    def predict(self, images):
        outputs = self.model(images)
        return outputs


def main(args):
    # data
    video_path = args.video
    frames = vid2frames(video_path)

    # batches 
    batches = batch_generator(frames, batch_size=16)
    batches_cp = batches.copy()

    # model
    weights_path = args.weights
    model = YOLO(weights_path)
    model_ref = ray.put(model)

    # ray actors
    N_ACTORS = 4
    idle_actors = []
    for i in range(N_ACTORS):
        idle_actors.append(
            PredictActor.remote(
                model=model_ref
            )
        )

    # batch inference using ray actors
    # predictions - list of final predictions
    # future_to_actor_mapping - a dictionary that maps ObjectReferences to the actor that promised them
    start = time.time()
    predictions = []
    future_to_actor_mapping = {}
    while batches:
        if idle_actors:
            actor = idle_actors.pop()
            batch = batches.pop()
            future = actor.predict.remote(images=batch)
            future_to_actor_mapping[future] = actor
        else:
            [ready], _ = ray.wait(list(future_to_actor_mapping.keys()), num_returns=1)
            actor = future_to_actor_mapping.pop(ready)
            idle_actors.append(actor)
            predictions = prediction_results_postprocessing(
                predictions=predictions, outs=ray.get(ready)
            )

    # Process any leftover results at the end.
    for future in future_to_actor_mapping.keys():
        predictions = prediction_results_postprocessing(
            predictions=predictions, outs=ray.get(future)
        )
    duration = time.time() - start
    
    print("--"*50)
    print("YOLOv8 Inference Duration: {}".format(duration))
    print("--"*50)


def get_args():
    ap = ArgumentParser()
    ap.add_argument("--video", default=os.path.join(os.getcwd(), "data", "videos", "demo.mp4"), help="Path to video")
    ap.add_argument("--weights", default=os.path.join(os.getcwd(), "yolo", "weights", "yolov8n.pt"), help="Path to YOLO weights")

    args = ap.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # ray initialization
    ray.init(address='auto') # ray start --head --port=6379
    time.sleep(1)

    # cluster configuration
    print("--"*50)
    print('''This cluster consists of
            {} nodes in total
            {} CPU resources in total'''.format(len(ray.nodes()), ray.cluster_resources()['CPU'])
        )
    print("--"*50)

    main(args)


