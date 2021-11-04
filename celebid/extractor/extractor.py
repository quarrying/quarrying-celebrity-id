import os

import cv2
import khandy
import numpy as np


def normalize_image_shape(image):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        num_channels = image.shape[-1]
        if num_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif num_channels == 3:
            pass
        elif num_channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError('Unsupported!')
    else:
        raise ValueError('Unsupported!')
    return image
    
    
class FaceFeatureExtractor(object):
    def __init__(self, model_filename=None):
        if model_filename is None:
            model_filename = os.path.join(os.path.dirname(__file__), 'models/model_ir_se50.onnx')
        self.model_filename = model_filename
        self.net = cv2.dnn.readNetFromONNX(model_filename)
        self.align_size = self.get_align_size()
        self.reference_landmarks = self.get_reference_landmarks()
        
    @staticmethod
    def get_align_size():
        return (112, 112)
        
    @staticmethod
    def get_reference_landmarks():
        # adapted from 112x96 standard landmarks
        std_landmarks = [[30.2946 + 8, 51.6963],  # left eye
                         [65.5318 + 8, 51.5014],  # right eye
                         [48.0252 + 8, 71.7366],  # nose tip
                         [33.5493 + 8, 92.3655],  # left mouth corner
                         [62.7299 + 8, 92.2041]]  # right mouth corner
        return std_landmarks
        
    def align_and_crop(self, image, landmarks):
        landmarks = np.asarray(landmarks).reshape(2, 5).T
        image_cropped, _ = khandy.align_and_crop(image, landmarks, 
                                                 self.reference_landmarks, 
                                                 self.align_size)
        return image_cropped
        
    @staticmethod
    def _preprocess(images):
        images = images.astype(np.float32)
        images -= 127.5
        images /= 127.5
        images = np.transpose(images, (0, 3, 1, 2))
        return images
        
    @staticmethod
    def get_feature_dim():
        return 512

    def extract(self, image):
        if image.ndim == 3:
            image = normalize_image_shape(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)
        image = self._preprocess(image)
        self.net.setInput(image)
        features = self.net.forward()
        features = khandy.l2_normalize(features, axis=-1)
        return features

        