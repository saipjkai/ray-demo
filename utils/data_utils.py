import random
random.seed(42)

import numpy as np
import cv2


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


def show_frames(imgs, window_name="stream"):
    if len(imgs) == 0:
        print("No images found!")
        exit(0)
    elif not isinstance(imgs[0], (np.ndarray, np.generic)):
        print("Unknown image type - NDArray not detected!")
        exit(0)

    for img in imgs:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(40) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p') :
            key = cv2.waitKey(0)
    cv2.destroyAllWindows()


def batch_generator(imgs, batch_size=1):
    n = len(imgs)
    img_indices = list(range(n))

    batches = []
    for batch_no in range(n//batch_size):
        batches.append(imgs[batch_no*batch_size:(batch_no+1)*batch_size])
    if (batch_no+1)*batch_size < n:
        batches.append(imgs[(batch_no+1)*batch_size:])        
    return batches
