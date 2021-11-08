import os
import time

import cv2
import khandy
import numpy as np


def imread_ex(filename, flags=-1):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
    except Exception as e:
        print('Image decode error!', e)
        return None
        

def imwrite_ex(filename, image):
    cv2.imencode(os.path.splitext(filename)[-1], image)[1].tofile(filename)
    
    
def clean_faces(src_dir, detector, dst_dir_prefix=None, min_face_size=30):
    """筛选出有人脸的图像 (保持文件原有内容)
    """
    src_dir = os.path.normpath(src_dir)
    dst_dir_prefix = dst_dir_prefix or src_dir
    dst_dir_prefix = os.path.normpath(dst_dir_prefix)
    
    dst_dir_nonface = dst_dir_prefix + ' nonface'
    dst_dir_corrupt = dst_dir_prefix + ' corrupt'
    dst_dir_small = dst_dir_prefix + ' small'
    dst_dir_multiple = dst_dir_prefix + ' multiple'
    
    filenames = khandy.get_all_filenames(src_dir)
    for k, name in enumerate(filenames):
        print('[{}/{}] {}'.format(k+1, len(filenames), name))
        start_time = time.time()
        
        img = imread_ex(name, 1)
        if (img is None) or (img.dtype != np.uint8):
            print('Image file corrupted!')
            khandy.move_file(name, dst_dir_corrupt)
            continue
        image_height, image_width = img.shape[:2]
        
        detected_rects, _ = detector.detect(img)
        max_face_size_str = ''
        if len(detected_rects) == 0: 
            khandy.move_file(name, dst_dir_nonface)
        elif len(detected_rects) > 1:
            khandy.move_file(name, dst_dir_multiple)
        else:
            detected_rects = khandy.clip_boxes_to_image(detected_rects, image_width, image_height, subpixel=False)
            max_face_rect = max(detected_rects, key=lambda rect: (rect[3] - rect[1]) * (rect[2] - rect[0]))
            max_face_width = int(round(max_face_rect[2] - max_face_rect[0] + 1))
            max_face_height = int(round(max_face_rect[3] - max_face_rect[1] + 1))
            max_face_size_str = '. Max face WxH: {}x{}'.format(max_face_width, max_face_height)
            if (max_face_width < min_face_size) or (max_face_height < min_face_size):
                khandy.move_file(name, dst_dir_small)

        print('Elapsed: {:.3f}s. Image Size: {}. Detected Face Number: {}{}'.format(
              time.time() - start_time, img.shape[:2], len(detected_rects), max_face_size_str))
    return src_dir
    

def crop_faces(src_dir, detector, dst_dir_prefix=None,
               width_scale=2.5, height_scale=2.5,
               extname='.jpg', only_max_face=False):
    """输入文件夹, 检测人脸并将之裁剪为大头照 (没有对齐)
    """
    src_dir = os.path.abspath(os.path.expanduser(src_dir))
    dst_dir_prefix = dst_dir_prefix or src_dir
    dst_dir_prefix = os.path.normpath(dst_dir_prefix)

    dst_dir_nonface = dst_dir_prefix + ' nonface'
    dst_dir_corrupt = dst_dir_prefix + ' corrupt'
    dst_dir_face = dst_dir_prefix + ' faces'
    dst_dir_multiple = dst_dir_prefix + ' multiple'

    filenames = khandy.get_all_filenames(src_dir)
    for k, name in enumerate(filenames):
        print('[{}/{}] {}'.format(k+1, len(filenames), name))
        start_time = time.time()

        img = imread_ex(name, -1)
        if (img is None) or (img.dtype != np.uint8):
            print('Image file corrupted!')
            khandy.move_file(name, dst_dir_corrupt)
            continue

        detected_rects, _ = detector.detect(img)
        if len(detected_rects) == 0:
            khandy.move_file(name, dst_dir_nonface)
        else:
            scaled_rects = khandy.scale_boxes_wrt_centers(detected_rects, width_scale, height_scale)
            scaled_rects = scaled_rects.astype(np.int32)
            if not only_max_face:
                keep = khandy.non_max_suppression(scaled_rects, scaled_rects[:, 4], 0.4)
                scaled_rects = scaled_rects[keep]
                stem, old_ext = os.path.splitext(os.path.basename(name))
                zfill_width = khandy.get_order_of_magnitude(len(scaled_rects)) + 1
                for k, scaled_rect in enumerate(scaled_rects):
                    cropped = khandy.crop_or_pad(img, scaled_rect[0], scaled_rect[1], scaled_rect[2], scaled_rect[3])
            
                    ordinal_str = '{}'.format(k).zfill(zfill_width)
                    os.makedirs(dst_dir_face, exist_ok=True)
                    dst_filename = os.path.join(dst_dir_face, '{}_{}{}'.format(stem, ordinal_str, old_ext))
                    dst_filename = khandy.replace_path_extension(dst_filename, extname)
                    imwrite_ex(dst_filename, cropped)
            else:
                max_face_rect = max(scaled_rects, key=lambda rect: (rect[3] - rect[1]) * (rect[2] - rect[0]))
                cropped = khandy.crop_or_pad(img, max_face_rect[0], max_face_rect[1], max_face_rect[2], max_face_rect[3])
                
                os.makedirs(dst_dir_face, exist_ok=True)
                dst_filename = os.path.join(dst_dir_face, os.path.basename(name))
                dst_filename = khandy.replace_path_extension(dst_filename, extname)
                imwrite_ex(dst_filename, cropped)
                
        print('Elapsed: {:.3f}s. Image Size: {}. Detected Face Number: {}'.format(
              time.time() - start_time, img.shape[:2], len(detected_rects)))
    return dst_dir_face


def crop_faces_video(filename, detector, dst_dir=None, 
                     sample_rate=1, width_scale=2.5, height_scale=2.5,
                     extname='.jpg', only_max_face=False):
    """输入视频文件, 检测人脸并将之裁剪为大头照 (没有对齐)
    """
    dst_dir = dst_dir or os.path.splitext(filename)[0] + '_faces'
    os.makedirs(dst_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(filename)
    
    image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('image_width: {}'.format(image_width))
    print('image_height: {}'.format(image_height))
    print('frame_rate: {}'.format(frame_rate))
    print('frame_count: {}'.format(frame_count))
    
    frame_zfill_width = khandy.get_order_of_magnitude(frame_count) + 1
    frame_no = 0
    while True:
        start_time = time.time()
        has_frame, img = cap.read()
        if not has_frame:
            break
        
        frame_no += 1
        if frame_no % sample_rate != 0:
            continue
            
        frame_no_str = '{}'.format(frame_no).zfill(frame_zfill_width)
        detected_rects, _ = detector.detect(img)

        if len(detected_rects) != 0:
            scaled_rects = khandy.scale_boxes_wrt_centers(detected_rects, width_scale, height_scale)
            scaled_rects = scaled_rects.astype(np.int32)
            if not only_max_face:
                keep = khandy.non_max_suppression(scaled_rects, scaled_rects[:, 4], 0.4)
                scaled_rects = scaled_rects[keep]
                face_zfill_width = khandy.get_order_of_magnitude(len(scaled_rects)) + 1
                for k, scaled_rect in enumerate(scaled_rects):
                    cropped = khandy.crop_or_pad(img, scaled_rect[0], scaled_rect[1], scaled_rect[2], scaled_rect[3])
                    
                    ordinal_str = '{}'.format(k).zfill(face_zfill_width)
                    dst_filename = os.path.join(dst_dir, '{}_{}.jpg'.format(frame_no_str, ordinal_str))
                    dst_filename = khandy.replace_path_extension(dst_filename, extname)
                    imwrite_ex(dst_filename, cropped)
            else:
                max_face_rect = max(scaled_rects, key=lambda rect: (rect[3] - rect[1]) * (rect[2] - rect[0]))
                cropped = khandy.crop_or_pad(img, max_face_rect[0], max_face_rect[1], max_face_rect[2], max_face_rect[3])
                
                dst_filename = os.path.join(dst_dir, '{}.jpg'.format(frame_no_str))
                dst_filename = khandy.replace_path_extension(dst_filename, extname)
                imwrite_ex(dst_filename, cropped)
            
        print('[{}/{}] Elapsed: {:.3f}s, Detected Face Number: {}'.format(
              frame_no_str, frame_count, time.time() - start_time, len(detected_rects)))
    return dst_dir
    
    