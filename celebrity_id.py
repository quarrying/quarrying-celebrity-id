import time
import glob

import cv2
import khandy
import numpy as np

from detector import FaceDetector
from extractor import FaceFeatureExtractor


def get_topk_labels_and_distances(probe_features, gallery_features, gallery_labels, k=5):
    if probe_features.ndim == 1:
        probe_features = np.expand_dims(probe_features, 0)
    distances = khandy.pairwise_distances(probe_features, gallery_features)
    topk_distances, topk_indices = khandy.find_topk(distances, k, axis=-1, largest=False)
    topk_labels = []
    print(topk_distances, topk_indices, distances[0][569])
    for one_topk_indices in topk_indices:
        one_topk_labels = [gallery_labels[i] for i in one_topk_indices]
        topk_labels.append(one_topk_labels)
    return topk_labels, topk_distances
    

def imread_ex(filename, flags=-1):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
    except Exception as e:
        print('Image decode error!', e)
        return None
        
        
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
    
    
def draw_rectangles(image, boxes):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    for i in range(x1.shape[0]):
        cv2.rectangle(image, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), 
                      (0,255,0), 1)
    return image
    
    
def draw_landmarks(image, landmarks):
    num_landmarks = landmarks.shape[1] // 2
    for i, landmark in enumerate(landmarks):
        for x, y in zip(landmark[:num_landmarks], landmark[num_landmarks:]):
            cv2.circle(image, (int(x), int(y)), 1, (0,255,0), -1, 8)
    return image
    

def _test_1vs1():
    filenames1 = glob.glob('F:\_Data\Celebrity\chinese\politican_chinese\曾荫权 faces/*.jpg')
    filenames2 = glob.glob('F:\_Data\Celebrity\chinese\star_chinese_A\阿杜 faces/*.jpg')
    
    image1 = imread_ex(filenames1[0], 1)
    face_boxes1, face_landmarks1, features1 = detect_align_and_extract(image1, detector, extractor)
    for k, filename in enumerate(filenames1):
        image2 = imread_ex(filename, 1)
        if (image2 is None) or (image2.dtype != np.uint8):
            print('Image file corrupted!')
            continue
        if min(image2.shape[:2]) > 1280:
            image2 = khandy.resize_image_short(image2, 1280)
            
        face_boxes2, face_landmarks2, features2 = detect_align_and_extract(image2, detector, extractor)
        distances = np.sum((features1[0] - features2[0])**2)
        print(distances)
        
        image2 = draw_rectangles(image2, face_boxes2)
        image2 = draw_landmarks(image2, face_landmarks2)
        cv2.imshow('image2', image2)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
    
    
if __name__ == '__main__':
    detector = FaceDetector()
    extractor = FaceFeatureExtractor()
    
    gallery_feature_dict = np.load('gallery/celebrity_features.npy', allow_pickle=True).item()
    gallery_labels, gallery_features = khandy.convert_feature_dict_to_array(gallery_feature_dict)
    filenames = glob.glob('F:\_Data\Celebrity\chinese\politican_chinese\曾荫权 faces/*.jpg')
    for k, filename in enumerate(filenames):
        image = imread_ex(filename, 1)
        if (image is None) or (image.dtype != np.uint8):
            print('Image file corrupted!')
            continue
        if min(image.shape[:2]) > 1280:
            image = khandy.resize_image_short(image, 1280)
            
        face_boxes, face_landmarks, labels, distances = compare_1vsn(image, gallery_features, gallery_labels, detector, extractor)
        print(labels, distances)
        
        image = draw_rectangles(image, face_boxes)
        image = draw_landmarks(image, face_landmarks)
        cv2.imshow('image', image)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()

    