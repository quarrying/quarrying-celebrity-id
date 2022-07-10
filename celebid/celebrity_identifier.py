import os

import khandy
import numpy as np

from .detector import FaceDetector
from .extractor import FaceFeatureExtractor


def get_topk_labels_and_distances(probe_features, gallery_features, gallery_labels, k=5):
    assert isinstance(k, int)
    if k <= 0:
        k = len(gallery_labels)
    k = min(len(gallery_labels), k)
    
    if probe_features.ndim == 1:
        probe_features = np.expand_dims(probe_features, 0)
    distances = khandy.pairwise_distances(probe_features, gallery_features)
    topk_distances, topk_indices = khandy.top_k(distances, k, axis=-1, largest=False)
    topk_labels = []
    for one_topk_indices in topk_indices:
        one_topk_labels = [gallery_labels[i] for i in one_topk_indices]
        topk_labels.append(one_topk_labels)
    return topk_labels, topk_distances
    

class CelebrityIdentifier(object):
    def __init__(self, size_thresh=40, conf_thresh=0.5, nms_thresh=0.5):
        curr_dir = os.path.dirname(__file__)
        gallery_feature_dict = np.load(os.path.join(curr_dir, 'celebrity_features.npy'), allow_pickle=True).item()
        self.gallery_labels, self.gallery_features = khandy.convert_feature_dict_to_array(gallery_feature_dict)
        self.detector = FaceDetector(conf_thresh=conf_thresh, nms_thresh=nms_thresh, size_thresh=size_thresh)
        self.extractor = FaceFeatureExtractor()
        
    def get_celebrity_names(self):
        return self.gallery_labels
        
    def detect_align_and_extract(self, image):
        face_boxes, face_scores, face_landmarks = self.detector.detect(image)
        feature_dim = self.extractor.get_feature_dim()
        features = np.empty((len(face_landmarks), feature_dim), np.float32)
        for i, face_landmark in enumerate(face_landmarks):
            aligned_face_image = self.extractor.align_and_crop(image, face_landmark)
            features[i] = self. extractor.extract(aligned_face_image)
        return face_boxes, face_scores, face_landmarks, features

    def identify(self, image, k=5):
        face_boxes, face_scores, face_landmarks, features = self.detect_align_and_extract(image)
        labels, distances = get_topk_labels_and_distances(features, self.gallery_features, self.gallery_labels, k)
        return face_boxes, face_landmarks, labels, distances
        

    