from .mtcnn import MTCNN


class FaceDetector(object):
    def __init__(self, model_dir=None):
        self.detector = MTCNN(model_dir)
        
    def detect(self, image, min_size=40, conf_thresholds=[0.6, 0.7, 0.7], factor=0.709):
        return self.detector.detect(image, min_size, conf_thresholds, factor)
        
        
