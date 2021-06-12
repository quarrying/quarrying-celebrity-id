import time
import glob

import cv2
import numpy as np
from mtcnn import MTCNN
# from mtcnn_raw import MTCNN


def imread_ex(filename, flags=-1):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
    except Exception as e:
        print('Image decode error!', e)
        return None
        
        
def draw_rectangles(image, boxes):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
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
    
    
if __name__ == '__main__':
    min_size = 20
    threshold = [0.6, 0.7, 0.7]
    factor = 0.709
    mtcnn = MTCNN()
    
    filenames = glob.glob(r'C:\Users\imkan\Desktop\faces\*.jpg')
    for k, filename in enumerate(filenames):
        print('[{}/{}]{}'.format(k+1, len(filenames), filename))
        img = imread_ex(filename)
        img_matlab = img.copy()
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp
        
        start_time = time.time()
        boxes, points = mtcnn.detect(img_matlab, min_size, threshold, factor)
        print(time.time() - start_time)
        print('num_faces: ', len(boxes))
        img = draw_rectangles(img, boxes)
        img = draw_landmarks(img, points)
        cv2.imshow('img', img)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
    
    