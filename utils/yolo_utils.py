import os
import cv2
from ultralytics import YOLO


def load_classes(classes_path):
    try:
        with open(classes_path, 'r') as f:
            classes_file_contents = f.read().strip().split('\n')
            classes = dict()
            for _ in classes_file_contents:
                class_id, class_name = _.split(',')
                classes[int(class_id)] = class_name
        return classes
    except:
        print("Classes file not found!")
        exit(0)


def draw_predictions(imgs, predictions, classes):
    outs = []
    for img, prediction in zip(imgs, predictions):
        # detections - bbox, conf, label
        bbox = prediction.boxes.xyxy        # box with xywh format, (N, 4)
        bbox_conf = prediction.boxes.conf   # confidence score, (N, 1)
        bbox_label = prediction.boxes.cls   # cls, (N, 1)
        
        # draw predictions
        for b, b_c, b_l in zip(bbox, bbox_conf, bbox_label):
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 1)
            cv2.putText(img, f"{classes[int(b_l)+1]}: {b_c:.2f}", (int(b[0]), int(b[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        outs.append(img)
    return outs


def save_results(imgs_list, output_dir):
    if len(imgs_list) == 1:
        output_path = os.path.join(output_dir, "result.jpg") 
        cv2.imwrite(output_path, imgs_list[0])
    elif len(imgs_list) > 1:
        output_path = os.path.join(output_dir, "result.avi")
        frame_height, frame_width, _ = imgs_list[0].shape
        result_video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 24, (frame_width, frame_height))
        for result in imgs_list:
            result_video_writer.write(result)
        result_video_writer.release()