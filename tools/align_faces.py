import os
import time

import cv2
import khandy
import numpy as np


def imread_ex(filename, flags=-1):
    """cv2.imread 的扩展, 使支持中文路径.
    """
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
    except Exception as e:
        # 当文件内容为空时, 可能会报出如下的错误
        # cv2.error: OpenCV(4.0.0) error: (-215:Assertion failed) !buf.empty() 
        # && buf.isContinuous() in function 'cv::imdecode_'
        print('Image decode error!', e)
        return None
        
        
def get_max_face_index(landmarks):
    landmarks = np.asarray(landmarks).reshape(-1, 2, 5)
    left_eye = landmarks[:, :, 0]
    right_eye = landmarks[:, :, 1]
    distance = np.sum((left_eye - right_eye) ** 2, axis=1)
    return np.argmax(distance)
    
    
def align_faces(src_dir, detector, reference_landmarks, align_size, 
                dst_dir=None, extname='.jpg'):
    src_dir = os.path.abspath(os.path.expanduser(src_dir))
    dst_dir = dst_dir or src_dir + '_crop'
    dst_dir = os.path.abspath(os.path.expanduser(dst_dir))
    os.makedirs(dst_dir, exist_ok=True)
    
    filenames = khandy.get_all_filenames(src_dir)
    for k, name in enumerate(filenames):
        print('[{}/{}] {}'.format(k+1, len(filenames), name))
        start_time = time.time()
        img = imread_ex(name, 1)
        if (img is None) or (img.dtype != np.uint8):
            print('Image file corrupted!')
            continue
        if min(img.shape[:2]) > 1280:
            img = khandy.resize_image_short(img, 1280)

        boxes, landmarks = detector.detect(img)
        if len(boxes) != 0:
            max_index = np.argmax(boxes[:, -1])
            # max_index = get_max_face_index(landmarks)
            max_landmarks = np.asarray(landmarks[max_index]).reshape(2, 5).T
            aligned, _ = khandy.align_and_crop(img, max_landmarks, reference_landmarks, align_size)
    
            dst_filename = name.replace(src_dir, dst_dir)
            dst_filename = khandy.replace_path_extension(dst_filename, extname)
            os.makedirs(os.path.dirname(dst_filename), exist_ok=True)
            cv2.imencode(os.path.splitext(dst_filename)[-1], aligned)[1].tofile(dst_filename)
            
