import os

import cv2
import khandy
import numpy as np

from .base import OnnxModel


class FaceDetector(OnnxModel):
    def __init__(self, model_filename=None):
        if model_filename is None:
            model_filename = os.path.join(os.path.dirname(__file__), 'models/retinaface.onnx')
        super(FaceDetector, self).__init__(model_filename)

        self.input_width = 512
        self.input_height = 512
        self.stddevs = [0.1, 0.1, 0.2, 0.2]
        self.anchor_sizes = [[16,16, 32,32], 
                             [64,64, 128,128], 
                             [256,256, 512,512]]
        self.strides = [8, 16, 32]
        self.priors = self.generate_priors(self.input_height, self.input_width, self.anchor_sizes, self.strides)
        
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
        # centers shape is (2, K), where K=fmap_height*fmap_width
        centers = np.vstack([xx_centers.flat, yy_centers.flat])
        # Shape is (K, 2)
        centers = centers.transpose()
        # Shape is (K, 1, 2)
        centers = centers.reshape((-1, 1, 2))
        
        # Shape is (Ax2,)
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
        # 注意有除以宽高
        prior_boxes /= np.array([image_width, image_height, image_width, image_height])
        return prior_boxes

    @staticmethod
    def decode_boxes(rel_boxes, reference_boxes, stddevs):
        reference_boxes = np.expand_dims(reference_boxes, 0)
        xcenters_ycenters = reference_boxes[..., 0:2] + rel_boxes[..., 0:2] * stddevs[0:2] * reference_boxes[..., 2:4]
        widths_heights = reference_boxes[..., 2:4] * np.exp(rel_boxes[..., 2:4] * stddevs[2:4])
        boxes = np.concatenate((xcenters_ycenters, widths_heights), -1)
        return boxes
        
    @staticmethod
    def decode_landmarks(rel_landmarks, reference_boxes, stddevs):
        batch_size, num_anchors = rel_landmarks.shape[:2]
        reference_boxes = reference_boxes.reshape(1, num_anchors, 1, -1)
        rel_landmarks = rel_landmarks.reshape(batch_size, num_anchors, -1, 2)
        landmarks = reference_boxes[..., 0:2] + rel_landmarks * stddevs[0:2] * reference_boxes[..., 2:4]
        return landmarks

    def preprocess(self, image):
        image_dtype = image.dtype
        assert image_dtype in [np.uint8, np.uint16]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image, scale, left, top = khandy.letterbox_resize_image(
            image, self.input_width, self.input_height, return_scale=True)
        image = image.astype(np.float32)
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image, axis=0)
        return image, scale, left, top

    def postprocess(self, pred_boxes, classes, pred_landmarks, scale, left, top, 
                    conf_thresh, nms_thresh, size_thresh, top_k):
        boxes = self.decode_boxes(pred_boxes, self.priors, self.stddevs)
        boxes = khandy.convert_boxes_format(boxes, 'cxcywh', 'xyxy', False)
        boxes[..., 0] = (boxes[..., 0] * self.input_width - left) / scale
        boxes[..., 1] = (boxes[..., 1] * self.input_height - top) / scale
        boxes[..., 2] = (boxes[..., 2] * self.input_width - left) / scale
        boxes[..., 3] = (boxes[..., 3] * self.input_height - top) / scale
        
        landmarks = self.decode_landmarks(pred_landmarks, self.priors, self.stddevs)
        landmarks[..., 0] = (landmarks[..., 0] * self.input_width - left) / scale
        landmarks[..., 1] = (landmarks[..., 1] * self.input_height - top) / scale

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
        
        # keep top-k before NMS
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

    def detect(self, image, conf_thresh=0.5, nms_thresh=0.5, size_thresh=30, top_k=None):
        img, scale, left, top = self.preprocess(image)
        # pred_boxes shape:     (batch_size, num_anchors, 4) 
        # classes shape:        (batch_size, num_anchors, 2) 
        # pred_landmarks shape: (batch_size, num_anchors, 10) 
        pred_boxes, classes, pred_landmarks = self.forward(img)

        boxes, scores, landmarks = self.postprocess(
            pred_boxes, classes, pred_landmarks, 
            scale=scale, left=left, top=top,
            conf_thresh=conf_thresh, nms_thresh=nms_thresh, 
            size_thresh=size_thresh, top_k=top_k)
        return boxes, scores, landmarks

