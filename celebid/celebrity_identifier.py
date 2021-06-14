import os
import sys

import khandy
import numpy as np

sys.path.insert(0, '..')
from detector import FaceDetector
from extractor import FaceFeatureExtractor


def get_topk_labels_and_distances(probe_features, gallery_features, gallery_labels, k=5):
    if probe_features.ndim == 1:
        probe_features = np.expand_dims(probe_features, 0)
    distances = khandy.pairwise_distances(probe_features, gallery_features)
    topk_distances, topk_indices = khandy.find_topk(distances, k, axis=-1, largest=False)
    topk_labels = []
    for one_topk_indices in topk_indices:
        one_topk_labels = [gallery_labels[i] for i in one_topk_indices]
        topk_labels.append(one_topk_labels)
    return topk_labels, topk_distances
    

def detect_align_and_extract(image, detector, extractor):
    face_boxes, face_landmarks = detector.detect(image)
    features = []
    for face_landmark in face_landmarks:
        aligned_face_image = extractor.align_and_crop(image, face_landmark)
        feature = extractor.extract(aligned_face_image)
        features.append(feature)
    features = np.vstack(features)
    return face_boxes, face_landmarks, features
    

def compare_1vsn(image, gallery_features, gallery_labels, detector, extractor):
    face_boxes, face_landmarks, features = detect_align_and_extract(image, detector, extractor)
    labels, distances = get_topk_labels_and_distances(features, gallery_features, gallery_labels)
    return face_boxes, face_landmarks, labels, distances
    
    
class CelebrityIdentifier(object):
    def __init__(self):
        curr_dir = os.path.dirname(__file__)
        gallery_feature_dict = np.load(os.path.join(curr_dir, 'celebrity_features.npy'), allow_pickle=True).item()
        self.gallery_labels, self.gallery_features = khandy.convert_feature_dict_to_array(gallery_feature_dict)
        self.detector = FaceDetector()
        self.extractor = FaceFeatureExtractor()
        
    def identify(self, image):
        face_boxes, face_landmarks, labels, distances = compare_1vsn(image, self.gallery_features, 
                                                                     self.gallery_labels, 
                                                                     self.detector, 
                                                                     self.extractor)
        return face_boxes, face_landmarks, labels, distances
        

    