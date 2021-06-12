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
    
    Args:
        boxes: (N, 4+K)
        width_delta: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
        height_delta: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
        copy: bool
        
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
    
    
def pad(boxesA, w, h):
    boxes = boxesA.copy()
    tmph = boxes[:,3] - boxes[:,1] + 1
    tmpw = boxes[:,2] - boxes[:,0] + 1
    numbox = boxes.shape[0]

    dx = np.ones(numbox)
    dy = np.ones(numbox)
    edx = tmpw 
    edy = tmph

    x = boxes[:,0:1][:,0]
    y = boxes[:,1:2][:,0]
    ex = boxes[:,2:3][:,0]
    ey = boxes[:,3:4][:,0]
   
    tmp = np.where(ex > w)[0]
    if tmp.shape[0] != 0:
        edx[tmp] = -ex[tmp] + w-1 + tmpw[tmp]
        ex[tmp] = w-1

    tmp = np.where(ey > h)[0]
    if tmp.shape[0] != 0:
        edy[tmp] = -ey[tmp] + h-1 + tmph[tmp]
        ey[tmp] = h-1

    tmp = np.where(x < 1)[0]
    if tmp.shape[0] != 0:
        dx[tmp] = 2 - x[tmp]
        x[tmp] = np.ones_like(x[tmp])

    tmp = np.where(y < 1)[0]
    if tmp.shape[0] != 0:
        dy[tmp] = 2 - y[tmp]
        y[tmp] = np.ones_like(y[tmp])
    
    # for python index from 0, while matlab from 1
    dy = np.maximum(0, dy-1)
    dx = np.maximum(0, dx-1)
    y = np.maximum(0, y-1)
    x = np.maximum(0, x-1)
    edy = np.maximum(0, edy-1)
    edx = np.maximum(0, edx-1)
    ey = np.maximum(0, ey-1)
    ex = np.maximum(0, ex-1)

    return [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    
    
def nms(boxes, threshold, type):
    if boxes.shape[0] == 0:
        return np.array([])
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
    def __init__(self):
        self.pnet = cv2.dnn.readNetFromCaffe('models/det1.prototxt', 'models/det1.caffemodel')
        self.rnet = cv2.dnn.readNetFromCaffe('models/det2.prototxt', 'models/det2.caffemodel')
        self.onet = cv2.dnn.readNetFromCaffe('models/det3.prototxt', 'models/det3.caffemodel')
        self.pnet_nms_thresh_intra = 0.5
        self.pnet_nms_thresh_inter = 0.7
        self.rnet_nms_thresh = 0.7
        self.onet_nms_thresh = 0.7

    @staticmethod
    def _get_scale_factors(min_side_length, factor, min_size):
        factor_count = 0
        m = 12.0 / min_size
        min_side_length = min_side_length * m
        scale_factors = []
        while min_side_length >= 12:
            scale_factors.append(m * pow(factor, factor_count))
            min_side_length *= factor
            factor_count += 1
        return scale_factors
    
    @staticmethod
    def _preprocess(images):
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
    def _decode_points(points, reference_boxes, copy=True):
        points = np.array(points, dtype=np.float32, copy=copy)
        reference_widths = reference_boxes[:, 2] - reference_boxes[:, 0] + 1
        reference_heights = reference_boxes[:, 3] - reference_boxes[:, 1] + 1
        points[:, 0:5] *= np.expand_dims(reference_widths, axis=1)
        points[:, 5:10] *= np.expand_dims(reference_heights, axis=1)
        points[:, 0:5] += np.expand_dims(reference_boxes[:, 0], axis=1) - 1 
        points[:, 5:10] += np.expand_dims(reference_boxes[:, 1], axis=1) - 1
        return points

    @staticmethod
    def _generate_boxes(cls_probs, loc_preds, scale_factor, conf_threshold):
        stride = 2
        cellsize = 12
        y, x = np.where(cls_probs >= conf_threshold)
        loc_offsets = np.array([loc_preds[0, :, :][y, x], 
                                loc_preds[1, :, :][y, x], 
                                loc_preds[2, :, :][y, x], 
                                loc_preds[3, :, :][y, x]])
        xy_coords = np.array([y, x]) # N.B.
        xy_mins = np.fix((stride * xy_coords + 1) / scale_factor)
        xy_maxs = np.fix((stride * xy_coords + cellsize - 1 + 1) / scale_factor)
        score = np.expand_dims(cls_probs[y, x], axis=0)
        results = np.concatenate((xy_mins, xy_maxs, score, loc_offsets), axis=0)
        return results.T
        
    @staticmethod
    def _crop_and_resize(img, dst_size, boxes):
        h, w = img.shape[:2]
        boxes = np.fix(boxes[:, :4])
        num_boxes = boxes.shape[0]
        dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(boxes, w, h)
        outputs = np.zeros((num_boxes, dst_size, dst_size, 3))
        for k in range(num_boxes):
            tmp = np.zeros((int(tmph[k]), int(tmpw[k]),3))
            tmp[int(dy[k]):int(edy[k])+1, int(dx[k]):int(edx[k])+1] = img[int(y[k]):int(ey[k])+1, int(x[k]):int(ex[k])+1]
            outputs[k,:,:,:] = cv2.resize(tmp, (dst_size, dst_size))
        return outputs

    def _run_first_stage(self, img, min_size, conf_threshold, factor):
        h, w = img.shape[:2]
        total_boxes = np.zeros((0, 9), np.float32)
        scale_factors = self._get_scale_factors(min(h, w), factor, min_size)
        for scale_factor in scale_factors:
            hs = int(np.ceil(h * scale_factor))
            ws = int(np.ceil(w * scale_factor))
            pnet_input = cv2.resize(img, (ws, hs))
            pnet_input = pnet_input.astype(np.float32)
            pnet_input = np.expand_dims(pnet_input, axis=0)
            pnet_input = self._preprocess(pnet_input)

            self.pnet.setInput(pnet_input)
            out_prob1, out_conv4_2 = self.pnet.forward(['prob1', 'conv4-2'])
            boxes = self._generate_boxes(out_prob1[0,1,:,:], out_conv4_2[0], 
                                         scale_factor, conf_threshold)
            if boxes.shape[0] != 0:
                pick = nms(boxes, self.pnet_nms_thresh_intra, 'Union')
                if len(pick) > 0 :
                    boxes = boxes[pick, :]
            if boxes.shape[0] != 0:
                total_boxes = np.concatenate((total_boxes, boxes), axis=0)
             
        if total_boxes.shape[0] != 0:
            pick = nms(total_boxes, self.pnet_nms_thresh_inter, 'Union')
            total_boxes = total_boxes[pick, :]
            score = np.expand_dims(total_boxes[:, 4], axis=1)
            total_boxes = self._decode_boxes(total_boxes[:,5:], total_boxes[:,:4])
            total_boxes = np.concatenate((total_boxes, score), axis=1)
            total_boxes = inflate_boxes_to_square(total_boxes)
        return total_boxes

    def detect(self, img, min_size, conf_thresholds, factor):
        img = normalize_image_shape(img)

        # First stage
        points = np.empty((0, 10), np.float32)
        total_boxes = self._run_first_stage(img, min_size, conf_thresholds[0], factor)
        if total_boxes.shape[0] == 0:
            return total_boxes, points

        # Second stage
        rnet_input = self._crop_and_resize(img, 24, total_boxes)
        rnet_input = self._preprocess(rnet_input)
        self.rnet.setInput(rnet_input)
        out_prob1, out_conv5_2 = self.rnet.forward(['prob1', 'conv5-2'])

        score = np.expand_dims(out_prob1[:, 1], axis=1)
        pass_t = np.where(score > conf_thresholds[1])[0]
        total_boxes = np.concatenate((total_boxes[pass_t, :4], score[pass_t, :]), axis=1)
        mv = out_conv5_2[pass_t, :]
        if total_boxes.shape[0] > 0:
            pick = nms(total_boxes, self.rnet_nms_thresh, 'Union')
            if len(pick) > 0 :
                total_boxes = self._decode_boxes(mv[pick, :], total_boxes[pick, :])
                total_boxes = inflate_boxes_to_square(total_boxes)
        if total_boxes.shape[0] == 0:
            return total_boxes, points

        # Third stage
        onet_input = self._crop_and_resize(img, 48, total_boxes)
        onet_input = self._preprocess(onet_input)
        self.onet.setInput(onet_input)
        out_prob1, out_conv6_2, out_conv6_3 = self.onet.forward(['prob1', 'conv6-2', 'conv6-3'])
        
        score = np.expand_dims(out_prob1[:, 1], axis=1)
        pass_t = np.where(score > conf_thresholds[2])[0]
        total_boxes = np.concatenate((total_boxes[pass_t, :4], score[pass_t, :]), axis=1)
        points = self._decode_points(out_conv6_3[pass_t, :], total_boxes)
        mv = out_conv6_2[pass_t, :]
        if total_boxes.shape[0] > 0:
            total_boxes = self._decode_boxes(mv, total_boxes)
            pick = nms(total_boxes, self.onet_nms_thresh, 'Min')
            if len(pick) > 0 :
                total_boxes = total_boxes[pick, :]
                points = points[pick, :]
        return total_boxes, points

