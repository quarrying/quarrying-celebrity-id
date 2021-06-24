import khandy
from .mtcnn import MTCNN


class FaceDetector(object):
    def __init__(self, model_dir=None, min_size=40, conf_thresholds=[0.6, 0.7, 0.7], factor=0.709):
        self.detector = MTCNN(model_dir)
        self.min_size = min_size
        self.conf_thresholds = conf_thresholds
        self.factor = factor
        self.max_side_length = 1280
        
    def detect(self, image):
        scale = 1
        if max(image.shape[:2]) > self.max_side_length:
            image, scale = khandy.resize_image_long(image, self.max_side_length, return_scale=True)
        face_boxes, face_landmarks = self.detector.detect(image, self.min_size, self.conf_thresholds, self.factor)
        face_boxes[:, :4] /= scale
        face_landmarks /= scale
        return face_boxes, face_landmarks 
        