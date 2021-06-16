from .mtcnn import MTCNN


class FaceDetector(object):
    def __init__(self, model_dir=None, min_size=40, conf_thresholds=[0.6, 0.7, 0.7], factor=0.709):
        self.detector = MTCNN(model_dir)
        self.min_size = min_size
        self.conf_thresholds = conf_thresholds
        self.factor = factor
        
    def detect(self, image):
        return self.detector.detect(image, self.min_size, self.conf_thresholds, self.factor)
        
        
