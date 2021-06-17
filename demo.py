import glob

import cv2
import khandy
import numpy as np

from celebid import CelebrityIdentifier


def imread_ex(filename, flags=-1):
    try:
        return cv2.imdecode(np.fromfile(filename, dtype=np.uint8), flags)
    except Exception as e:
        print('Image decode error!', e)
        return None
        
        
def draw_rectangles(image, boxes):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
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
    celeb_identifier = CelebrityIdentifier(min_size=40)
    filenames = glob.glob('F:/_Data/Celebrity/chinese/star_chinese_H/何泓姗 faces/*.jpg')
    for k, filename in enumerate(filenames):
        image = imread_ex(filename, 1)
        if (image is None) or (image.dtype != np.uint8):
            print('Image file corrupted!')
            continue
        if min(image.shape[:2]) > 1280:
            image = khandy.resize_image_short(image, 1280)
            
        face_boxes, face_landmarks, labels, distances = celeb_identifier.identify(image)
        print(labels, distances)
        
        image = draw_rectangles(image, face_boxes)
        image = draw_landmarks(image, face_landmarks)
        cv2.imshow('image', image)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
