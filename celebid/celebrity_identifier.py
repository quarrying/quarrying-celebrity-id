import os

import khandy
import numpy as np

from .detector import FaceDetector
from .extractor import FaceFeatureExtractor


def get_topk_labels_and_distances(probe_features, gallery_features, gallery_labels, k=5):
    if probe_features.ndim == 1:
        probe_features = np.expand_dims(probe_features, 0)
    distances = khandy.pairwise_distances(probe_features, gallery_features)
    topk_distances, topk_indices = khandy.top_k(distances, k, axis=-1, largest=False)
    topk_labels = []
    for one_topk_indices in topk_indices:
        one_topk_labels = [gallery_labels[i] for i in one_topk_indices]
        topk_labels.append(one_topk_labels)
    return topk_labels, topk_distances
    

def detect_align_and_extract(image, detector, extractor):
    face_boxes, face_landmarks = detector.detect(image)
    feature_dim = extractor.get_feature_dim()
    features = np.empty((len(face_landmarks), feature_dim), np.float32)
    for k, face_landmark in enumerate(face_landmarks):
        aligned_face_image = extractor.align_and_crop(image, face_landmark)
        features[k] = extractor.extract(aligned_face_image)
    return face_boxes, face_landmarks, features
    
    
class CelebrityIdentifier(object):
    def __init__(self, min_size=40):
        curr_dir = os.path.dirname(__file__)
        gallery_feature_dict = np.load(os.path.join(curr_dir, 'celebrity_features.npy'), allow_pickle=True).item()
        self.gallery_labels, self.gallery_features = khandy.convert_feature_dict_to_array(gallery_feature_dict)
        self.detector = FaceDetector(min_size=min_size)
        self.extractor = FaceFeatureExtractor()
        
    def get_celebrity_names(self):
        return self.gallery_labels
        
    def identify(self, image, k=5):
        face_boxes, face_landmarks, features = detect_align_and_extract(image, self.detector, self.extractor)
        labels, distances = get_topk_labels_and_distances(features, self.gallery_features, self.gallery_labels, k)
        return face_boxes, face_landmarks, labels, distances
        

    