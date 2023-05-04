import os

import khandy
import numpy as np

from .detector import FaceDetector
from .extractor import FaceFeatureExtractor


def get_topk(probe_features: np.ndarray, gallery_features: np.ndarray, k: int = 5):
    assert isinstance(k, int)
    if k <= 0:
        k = len(gallery_features)
    k = min(len(gallery_features), k)
    
    if probe_features.ndim == 1:
        probe_features = np.expand_dims(probe_features, 0)
    distances = khandy.pairwise_distances(probe_features, gallery_features)
    topk_distances, topk_indices = khandy.top_k(distances, k, axis=-1, largest=False)
    return topk_distances, topk_indices
    

def translate_labels(labels, label_map):
    dst_labels = []
    for item in labels:
        if isinstance(item, (list, tuple, np.ndarray)):
            dst_item = [label_map[i] for i in item]
        elif isinstance(item, (int, str)):
            dst_item = label_map[item]
        else:
            raise ValueError(f'Unsupported type: {type(item)}')
        dst_labels.append(dst_item)
    return dst_labels
    

class CelebrityIdentifier(object):
    def __init__(self, size_thresh=40, conf_thresh=0.5, nms_thresh=0.5):
        self.detector = FaceDetector(conf_thresh=conf_thresh, nms_thresh=nms_thresh, size_thresh=size_thresh)
        self.extractor = FaceFeatureExtractor()
        self.load_gallery()

    def load_gallery(self, gallery_filename=None):
        if gallery_filename is None:
            gallery_filename = os.path.join(os.path.dirname(__file__), 'models/celebrity_features.npy')
        gallery_feature_dict = np.load(gallery_filename, allow_pickle=True).item()
        self.gallery_labels, self.gallery_features = khandy.convert_feature_dict_to_array(gallery_feature_dict)

    def get_celebrity_names(self):
        return self.gallery_labels
        
    def detect_align_and_extract(self, image):
        face_boxes, face_scores, face_landmarks = self.detector.detect(image)
        feature_dim = self.extractor.get_feature_dim()
        features = np.empty((len(face_landmarks), feature_dim), np.float32)
        for i, face_landmark in enumerate(face_landmarks):
            aligned_face_image = self.extractor.align_and_crop(image, face_landmark)
            features[i] = self.extractor.extract(aligned_face_image)
        return face_boxes, face_scores, face_landmarks, features

    def identify(self, image, k=5):
        face_boxes, face_scores, face_landmarks, features = self.detect_align_and_extract(image)
        distances, indices = get_topk(features, self.gallery_features, k)
        labels = translate_labels(indices, self.gallery_labels)
        return face_boxes, face_landmarks, labels, distances
        
