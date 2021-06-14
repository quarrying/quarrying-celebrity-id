import os
import sys
import glob
from collections import OrderedDict

import cv2
import khandy
import numpy as np

sys.path.insert(0, '..')
from detector import FaceDetector
from extractor import FaceFeatureExtractor
from tools.align_faces import align_faces


def imread_ex(filename, flags=-1):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
    except Exception as e:
        print('Image decode error!', e)
        return None
        
        
if __name__ == '__main__':
    detector = FaceDetector()
    extractor = FaceFeatureExtractor()
    reference_landmarks = extractor.get_reference_landmarks()
    align_size = extractor.get_align_size()
    
    dst_dir_root = 'F:/_Data/Celebrity/_gallery_align/'
    # dirnames = khandy.get_top_level_dirs('F:/_Data/Celebrity/_gallery/')
    # for k, dirname in enumerate(dirnames):
    #     print('[{}/{}] {}'.format(k+1, len(dirnames), dirname))
    #     dst_dir = os.path.join(dst_dir_root, os.path.basename(dirname))
    #     align_faces(dirname, detector, reference_landmarks, align_size, dst_dir)
    
    feature_dict = OrderedDict()
    image_filenames = khandy.get_all_filenames(dst_dir_root)
    for k, image_filename in enumerate(image_filenames):
        print('[{}/{}] {}'.format(k+1, len(image_filenames), image_filename))
        image = imread_ex(image_filename, 1)
        if (image is None) or (image.dtype != np.uint8):
            print('Image file corrupted!')
            continue

        feature = extractor.extract(image)
        stem = khandy.get_path_stem(image_filename)
        feature_dict[stem] = feature
    parent_dir = os.path.dirname(os.path.dirname(__file__))
    np.save(os.path.join(parent_dir, 'gallery/celebrity_features.npy'), feature_dict)
    
    