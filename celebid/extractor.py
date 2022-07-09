import os

import cv2
import khandy
import numpy as np

from .base import OnnxModel


class FaceFeatureExtractor(OnnxModel):
    def __init__(self, model_filename=None):
        if model_filename is None:
            model_filename = os.path.join(os.path.dirname(__file__), 'models/model_ir_se50.onnx')
        super(FaceFeatureExtractor, self).__init__(model_filename)
        self.align_size = self.get_align_size()
        self.reference_landmarks = self.get_reference_landmarks()
        
    @staticmethod
    def get_align_size():
        return (112, 112)

    @staticmethod
    def get_feature_dim():
        return 512

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
        
    def extract(self, image):
        if image.ndim == 3:
            image = khandy.normalize_image_shape(image, swap_rb=True)
            image = np.expand_dims(image, axis=0)
        image = self._preprocess(image)
        features = self.forward(image)
        features = khandy.l2_normalize(features, axis=-1)
        return features

        