import os
import cv2
import numpy as np


def normalize_image_shape(image):
    """归一化到三通道二维图像
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        num_channels = image.shape[-1]
        if num_channels == 1:
            gray = np.squeeze(image, -1)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif num_channels == 3:
            pass
        elif num_channels == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError('Unsupported!')
    else:
        raise ValueError('Unsupported!')
    return image
    
    
def inflate_boxes_to_square(boxes, copy=True):
    """Inflate boxes to square
    
    References:
        `rerec` in https://github.com/kpzhang93/MTCNN_face_detection_alignment
    """
    boxes = np.array(boxes, dtype=np.float32, copy=copy)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    max_side_lengths = np.maximum(widths, heights)
    
    width_deltas = np.subtract(max_side_lengths, widths, widths)
    height_deltas = np.subtract(max_side_lengths, heights, heights)
    width_deltas *= 0.5
    height_deltas *= 0.5
    boxes[:, 0] -= width_deltas
    boxes[:, 1] -= height_deltas
    boxes[:, 2] += width_deltas
    boxes[:, 3] += height_deltas
    return boxes
    

def crop_or_pad_coords(boxes, image_width, image_height):
    x_mins = boxes[:, 0]
    y_mins = boxes[:, 1]
    x_maxs = boxes[:, 2]
    y_maxs = boxes[:, 3]
    dst_widths = x_maxs - x_mins + 1
    dst_heights = y_maxs - y_mins + 1

    src_x_begin = np.maximum(x_mins, 0)
    src_y_begin = np.maximum(y_mins, 0)
    src_x_end = np.minimum(x_maxs + 1, image_width)
    src_y_end = np.minimum(y_maxs + 1, image_height)
    
    dst_x_begin = np.maximum(-x_mins, 0)
    dst_y_begin = np.maximum(-y_mins, 0)
    dst_x_end = np.minimum(dst_widths, image_width - x_mins)
    dst_y_end = np.minimum(dst_heights, image_height - y_mins)

    return [dst_y_begin, dst_y_end, dst_x_begin, dst_x_end, 
            src_y_begin, src_y_end, src_x_begin, src_x_end, 
            dst_widths, dst_heights]
    
    
def nms(boxes, threshold, type):
    if boxes.shape[0] == 0:
        return []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = np.multiply(x2 - x1 + 1, y2 - y1 + 1)
    I = np.array(s.argsort())
    
    pick = [];
    while len(I) > 0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'Min':
            overlap_ratio = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            overlap_ratio = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(overlap_ratio <= threshold)[0]]
    return pick
    
    
class MTCNN(object):
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), 'models')
        self.pnet = cv2.dnn.readNetFromCaffe(os.path.join(model_dir, 'det1.prototxt'), 
                                             os.path.join(model_dir, 'det1.caffemodel'))
        self.rnet = cv2.dnn.readNetFromCaffe(os.path.join(model_dir, 'det2.prototxt'), 
                                             os.path.join(model_dir, 'det2.caffemodel'))
        self.onet = cv2.dnn.readNetFromCaffe(os.path.join(model_dir, 'det3.prototxt'), 
                                             os.path.join(model_dir, 'det3.caffemodel'))
        self.pnet_nms_thresh_intra = 0.5
        self.pnet_nms_thresh_inter = 0.7
        self.rnet_nms_thresh = 0.7
        self.onet_nms_thresh = 0.7

    @staticmethod
    def _get_scale_factors(min_side_length, factor, min_size):
        factor_count = 0
        min_detection_size = 12
        m = min_detection_size / min_size
        min_side_length *= m
        scale_factors = []
        while min_side_length >= min_detection_size:
            scale_factors.append(m * pow(factor, factor_count))
            min_side_length *= factor
            factor_count += 1
        return scale_factors
    
    @staticmethod
    def _preprocess(images):
        images = images.astype(np.float32)
        images -= 127.5
        images *= 0.0078125
        images = np.swapaxes(images, 1, 3)
        return images
        
    @staticmethod
    def _decode_boxes(boxes, reference_boxes, copy=True):
        boxes = np.array(boxes, dtype=np.float32, copy=copy)
        reference_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + 1
        reference_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + 1
        boxes[:, 0] *= reference_widths
        boxes[:, 1] *= reference_heights
        boxes[:, 2] *= reference_widths
        boxes[:, 3] *= reference_heights
        boxes[:, :4] += reference_boxes[:, :4]
        boxes = np.concatenate([boxes, reference_boxes[:, 4:]], axis=1)
        return boxes

    @staticmethod
    def _decode_landmarks(landmarks, reference_boxes, copy=True):
        landmarks = np.array(landmarks, dtype=np.float32, copy=copy)
        reference_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + 1
        reference_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + 1
        landmarks[:, 0:5] *= np.expand_dims(reference_widths, axis=1)
        landmarks[:, 5:10] *= np.expand_dims(reference_heights, axis=1)
        landmarks[:, 0:5] += np.expand_dims(reference_boxes[:, 0], axis=1) - 1 
        landmarks[:, 5:10] += np.expand_dims(reference_boxes[:, 1], axis=1) - 1
        return landmarks

    @staticmethod
    def _generate_boxes(cls_probs, loc_preds, scale_factor, conf_threshold):
        stride = 2
        cell_size = 12
        y, x = np.where(cls_probs >= conf_threshold)
        scores = np.expand_dims(cls_probs[y, x], axis=0)
        loc_offsets = np.array([loc_preds[0, y, x], 
                                loc_preds[1, y, x], 
                                loc_preds[2, y, x], 
                                loc_preds[3, y, x]])
        xy_coords = np.array([y, x]) # N.B.: not np.array([x, y])
        xy_mins = np.fix((stride * xy_coords + 1) / scale_factor)
        xy_maxs = np.fix((stride * xy_coords + cell_size - 1 + 1) / scale_factor)
        results = np.concatenate((xy_mins, xy_maxs, scores, loc_offsets), axis=0)
        return results.T
        
    @staticmethod
    def _crop_and_resize(image, dst_size, boxes):
        image_height, image_width = image.shape[:2]
        boxes = np.fix(boxes[:, :4]).astype(np.int32)
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = crop_or_pad_coords(boxes, image_width, image_height)
        outputs = np.empty((boxes.shape[0], dst_size, dst_size, 3), dtype=image.dtype)
        for k in range(boxes.shape[0]):
            tmp = np.zeros((tmph[k], tmpw[k], 3), dtype=image.dtype)
            tmp[dy[k]:edy[k], dx[k]:edx[k]] = image[y[k]:ey[k], x[k]:ex[k]]
            outputs[k, :, :, :] = cv2.resize(tmp, (dst_size, dst_size))
        return outputs

    def _run_first_stage(self, image, min_size, conf_threshold, factor):
        image_height, image_width = image.shape[:2]
        scale_factors = self._get_scale_factors(min(image_height, image_width), factor, min_size)
        boxes_list = []
        for scale_factor in scale_factors:
            dst_width = int(np.ceil(image_width * scale_factor))
            dst_height = int(np.ceil(image_height * scale_factor))
            pnet_input = cv2.resize(image, (dst_width, dst_height))
            pnet_input = np.expand_dims(pnet_input, axis=0)
            pnet_input = self._preprocess(pnet_input)

            self.pnet.setInput(pnet_input)
            out_prob1, out_conv4_2 = self.pnet.forward(['prob1', 'conv4-2'])
            boxes = self._generate_boxes(out_prob1[0, 1, :, :], out_conv4_2[0], 
                                         scale_factor, conf_threshold)
            pick = nms(boxes, self.pnet_nms_thresh_intra, 'Union')
            boxes_list.append(boxes[pick, :])

        boxes_ex = np.vstack(boxes_list)
        pick = nms(boxes_ex, self.pnet_nms_thresh_inter, 'Union')
        boxes = self._decode_boxes(boxes_ex[pick, 5:], boxes_ex[pick, :4])
        boxes = np.concatenate((boxes, boxes_ex[pick, 4:5]), axis=1)
        boxes = inflate_boxes_to_square(boxes)
        return boxes

    def detect(self, image, min_size=20, conf_thresholds=[0.6, 0.7, 0.7], factor=0.709):
        image = normalize_image_shape(image)

        # First stage
        landmarks = np.empty((0, 10), np.float32)
        boxes = self._run_first_stage(image, min_size, conf_thresholds[0], factor)
        if boxes.shape[0] == 0:
            return boxes, landmarks

        # Second stage
        rnet_input = self._crop_and_resize(image, 24, boxes)
        rnet_input = self._preprocess(rnet_input)
        self.rnet.setInput(rnet_input)
        out_prob1, out_conv5_2 = self.rnet.forward(['prob1', 'conv5-2'])

        scores = np.expand_dims(out_prob1[:, 1], axis=1)
        pass_t = np.where(scores > conf_thresholds[1])[0]
        boxes = np.concatenate((boxes[pass_t, :4], scores[pass_t, :]), axis=1)
        offsets = out_conv5_2[pass_t, :]

        pick = nms(boxes, self.rnet_nms_thresh, 'Union')
        boxes = self._decode_boxes(offsets[pick, :], boxes[pick, :])
        boxes = inflate_boxes_to_square(boxes)
        if boxes.shape[0] == 0:
            return boxes, landmarks

        # Third stage
        onet_input = self._crop_and_resize(image, 48, boxes)
        onet_input = self._preprocess(onet_input)
        self.onet.setInput(onet_input)
        out_prob1, out_conv6_2, out_conv6_3 = self.onet.forward(['prob1', 'conv6-2', 'conv6-3'])
        
        scores = np.expand_dims(out_prob1[:, 1], axis=1)
        pass_t = np.where(scores > conf_thresholds[2])[0]
        boxes = np.concatenate((boxes[pass_t, :4], scores[pass_t, :]), axis=1)
        landmarks = self._decode_landmarks(out_conv6_3[pass_t, :], boxes)
        offsets = out_conv6_2[pass_t, :]

        boxes = self._decode_boxes(offsets, boxes)
        pick = nms(boxes, self.onet_nms_thresh, 'Min')
        boxes = boxes[pick, :]
        landmarks = landmarks[pick, :]
        return boxes, landmarks

