import os

import cv2
import khandy
import numpy as np

from .base import OnnxModel
from .base import check_image_dtype_and_shape


class FaceDetector(OnnxModel):
    def __init__(self, model_filename=None, conf_thresh=0.5, nms_thresh=0.5, size_thresh=30, top_k=None):
        if model_filename is None:
            model_filename = os.path.join(os.path.dirname(__file__), 'models/retinaface.onnx')
        super(FaceDetector, self).__init__(model_filename)
        self._conf_thresh = conf_thresh
        self._nms_thresh = nms_thresh
        self._size_thresh = size_thresh
        self._top_k = top_k
        
        self.input_width = 512
        self.input_height = 512
        self.stddevs = [0.1, 0.1, 0.2, 0.2]
        self.anchor_sizes = [[16,16, 32,32], 
                             [64,64, 128,128], 
                             [256,256, 512,512]]
        self.strides = [8, 16, 32]
        self.priors = self.generate_priors(self.input_height, self.input_width, self.anchor_sizes, self.strides)
        self.boxes_coder = khandy.FasterRcnnBoxCoder(self.stddevs)

    @property
    def conf_thresh(self):
        return self._conf_thresh
        
    @conf_thresh.setter
    def conf_thresh(self, value):
        self._conf_thresh = value
        
    @property
    def nms_thresh(self):
        return self._nms_thresh
        
    @nms_thresh.setter
    def nms_thresh(self, value):
        self._nms_thresh = value
        
    @property
    def size_thresh(self):
        return self._size_thresh
        
    @size_thresh.setter
    def size_thresh(self, value):
        self._size_thresh = value

    @property
    def top_k(self):
        return self._top_k
        
    @top_k.setter
    def top_k(self, value):
        self._top_k = value
        
    @staticmethod
    def generate_grid_anchors(fmap_size, anchor_sizes, stride):
        fmap_height, fmap_width = fmap_size
        
        # x_centers shape is (fmap_width,)
        # y_centers shape is (fmap_height,)
        x_centers = (np.arange(fmap_width) + 0.5) * stride
        y_centers = (np.arange(fmap_height) + 0.5) * stride
        # xx_centers shape is (fmap_height, fmap_width)
        # yy_centers shape is (fmap_height, fmap_width)
        xx_centers, yy_centers = np.meshgrid(x_centers, y_centers)
        # centers shape is (2, K), where K = fmap_height * fmap_width
        centers = np.vstack([xx_centers.flat, yy_centers.flat])
        # Shape is (K, 2)
        centers = centers.transpose()
        # Shape is (K, 1, 2)
        centers = centers.reshape((-1, 1, 2))
        
        # Shape is (Ax2,), where A is num_base_anchors
        anchor_sizes = np.asarray(anchor_sizes)
        # Shape is (1, A, 2)
        anchor_sizes = anchor_sizes.reshape((1, -1, 2))
        # Shape is (K, A, 2)
        centers, anchor_sizes = np.broadcast_arrays(centers, anchor_sizes)
        # Shape is (K, A, 4)
        all_anchors = np.concatenate([centers, anchor_sizes], axis=-1)
        # Shape is (KxA, 4)
        all_anchors = all_anchors.reshape(-1, 4)
        return all_anchors

    @staticmethod
    def generate_priors(image_height, image_width, anchor_sizes, strides):
        anchors = []
        for k in range(len(anchor_sizes)):
            fmap_size = [(image_height + strides[k] - 1) // strides[k], 
                         (image_width + strides[k] - 1) // strides[k]]
            anchors.append(FaceDetector.generate_grid_anchors(fmap_size, anchor_sizes[k], strides[k]))
        prior_boxes = np.vstack(anchors)
        # NB: divided by image_width or image_height
        prior_boxes /= np.array([image_width, image_height, image_width, image_height])
        return prior_boxes

    def preprocess(self, image):
        check_image_dtype_and_shape(image)

        # image size normalization
        image, scale, pad_left, pad_top = khandy.letterbox_resize_image(
            image, self.input_width, self.input_height, return_scale=True)
        # image channel normalization
        image = khandy.normalize_image_channel(image, swap_rb=True)
        # image dtype normalization
        image_dtype = image.dtype
        image = image.astype(np.float32)
        if image_dtype == np.uint16:
            image /= 255.0
        # to tensor
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        return image, scale, pad_left, pad_top

    def postprocess(self, pred_boxes, classes, pred_landmarks, scale, pad_left, pad_top, 
                    conf_thresh, nms_thresh, size_thresh, top_k):
        boxes = self.boxes_coder.decode(pred_boxes, self.priors, False)
        boxes = khandy.convert_boxes_format(boxes, 'cxcywh', 'xyxy', False)
        boxes = khandy.scale_2d_points(boxes, self.input_width, self.input_height, False)
        boxes = khandy.unletterbox_2d_points(boxes, scale, pad_left, pad_top, False)

        landmarks = self.boxes_coder.decode_points(pred_landmarks, self.priors, False)
        landmarks = khandy.scale_2d_points(landmarks, self.input_width, self.input_height, False)
        landmarks = khandy.unletterbox_2d_points(landmarks, scale, pad_left, pad_top, False)
        landmarks = landmarks.reshape(*pred_landmarks.shape[:2], -1, 2)

        boxes = boxes.squeeze(0)
        landmarks = landmarks.squeeze(0)
        scores = classes.squeeze(0)[:, 1]

        # ignore low scores
        inds = np.nonzero(scores > conf_thresh)[0]
        boxes = boxes[inds]
        landmarks = landmarks[inds]
        scores = scores[inds]

        # ignore small size
        if size_thresh is not None:
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            shorter_sides = np.minimum(widths, heights)
            indexs = np.nonzero(shorter_sides > size_thresh)[0]
            boxes = boxes[indexs]
            landmarks = landmarks[indexs]
            scores = scores[indexs]
        
        # keep pad_top-k before NMS
        if top_k is not None:
            order = scores.argsort()[::-1][:top_k]
            boxes = boxes[order]
            landmarks = landmarks[order]
            scores = scores[order]

        # do NMS
        keep = khandy.non_max_suppression(boxes, scores, nms_thresh)
        boxes = boxes[keep]
        scores = scores[keep]
        landmarks = landmarks[keep]
        return boxes, scores, landmarks

    def detect(self, image):
        image, scale, pad_left, pad_top = self.preprocess(image)
        # pred_boxes shape:     (batch_size, num_anchors, 4) 
        # classes shape:        (batch_size, num_anchors, 2) 
        # pred_landmarks shape: (batch_size, num_anchors, 10) 
        pred_boxes, classes, pred_landmarks = self.forward(image)

        boxes, scores, landmarks = self.postprocess(
            pred_boxes, classes, pred_landmarks, 
            scale=scale, pad_left=pad_left, pad_top=pad_top,
            conf_thresh=self.conf_thresh, nms_thresh=self.nms_thresh, 
            size_thresh=self.size_thresh, top_k=self.top_k)
        return boxes, scores, landmarks

