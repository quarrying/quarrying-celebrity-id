import os
import time

import cv2
import khandy
import numpy as np


def align_and_crop_112x112(image, landmarks):
    align_size = (112, 112)
    # adapted from 112x96 standard landmarks
    std_landmarks = [[30.2946 + 8, 51.6963],  # left eye
                     [65.5318 + 8, 51.5014],  # right eye
                     [48.0252 + 8, 71.7366],  # nose tip
                     [33.5493 + 8, 92.3655],  # left mouth corner
                     [62.7299 + 8, 92.2041]]  # right mouth corner
    landmarks = np.asarray(landmarks).reshape(2, 5).T
    image_cropped, _ = khandy.align_and_crop(image, landmarks, std_landmarks, align_size)
    return image_cropped
    
    
def get_max_face_index(landmarks):
    landmarks = np.asarray(landmarks).reshape(-1, 2, 5)
    left_eye = landmarks[:, :, 0]
    right_eye = landmarks[:, :, 1]
    distance = np.sum((left_eye - right_eye) ** 2, axis=1)
    return np.argmax(distance)
    
    
def align_faces(src_dir, detector, dst_dir=None, extname='.jpg'):
    src_dir = os.path.abspath(os.path.expanduser(src_dir))
    dst_dir = dst_dir or src_dir + '_crop'
    dst_dir = os.path.abspath(os.path.expanduser(dst_dir))
    os.makedirs(dst_dir, exist_ok=True)
    
    filenames = khandy.get_all_filenames(src_dir)
    for k, name in enumerate(filenames):
        print('[{}/{}] {}'.format(k+1, len(filenames), name))
        start_time = time.time()
        try:
            img = cv2.imdecode(np.fromfile(name, dtype=np.uint8), 1)
        except Exception as e:
            print('Image decode error!', e)
            continue
        if (img is None) or (img.dtype != np.uint8):
            print('Image file corrupted!')
            continue
        if min(img.shape[:2]) > 1280:
            img = khandy.resize_image_short(img, 1280)

        boxes, landmarks = detector.detect(img)
        if len(boxes) != 0:
            max_index = np.argmax(boxes[:, -1])
            # max_index = get_max_face_index(landmarks)
            aligned = align_and_crop_112x112(img, landmarks[max_index])
            
            dst_filename = name.replace(src_dir, dst_dir)
            dst_filename = khandy.replace_path_extension(dst_filename, extname)
            os.makedirs(os.path.dirname(dst_filename), exist_ok=True)
            cv2.imencode(os.path.splitext(dst_filename)[-1], aligned)[1].tofile(dst_filename)
            
