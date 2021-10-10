import glob

import cv2
import khandy
import numpy as np
import PIL
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

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
                      (0,255,0), 2)
    return image
    
    
def draw_landmarks(image, landmarks):
    num_landmarks = landmarks.shape[1] // 2
    for i, landmark in enumerate(landmarks):
        for x, y in zip(landmark[:num_landmarks], landmark[num_landmarks:]):
            cv2.circle(image, (int(x), int(y)), 1, (0,255,0), -1, 8)
    return image
    
    
def draw_text(image, text, position, font_size=15, color=(255,0,0),
              font_filename='data/simsun.ttc'):
    assert isinstance(color, (tuple, list)) and len(color) == 3
    gray = color[0]
    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image)
        color = (color[2], color[1], color[0])
    elif isinstance(image, PIL.Image.Image):
        pil_image = image
    else:
        raise ValueError('Unsupported image type!')
    assert pil_image.mode in ['L', 'RGB']
    if pil_image.mode == 'L':
        color = gray

    font_object = ImageFont.truetype(font_filename, size=font_size)
    drawable = ImageDraw.Draw(pil_image)
    drawable.text((position[0], position[1]), text, 
                  fill=color, font=font_object)

    if isinstance(image, np.ndarray):
        return np.asarray(pil_image)
    return pil_image


if __name__ == '__main__':
    celeb_identifier = CelebrityIdentifier(min_size=40)
    filenames = glob.glob('F:/_Data/Celebrity/chinese/star_chinese_H/何泓姗 faces/*.jpg')
    for k, filename in enumerate(filenames):
        print('[{}/{}] {}'.format(k+1, len(filenames), filename))
        image = imread_ex(filename, 1)
        if (image is None) or (image.dtype != np.uint8):
            print('Image file corrupted!')
            continue
        if min(image.shape[:2]) > 1920:
            image = khandy.resize_image_short(image, 1920)
            
        face_boxes, face_landmarks, labels, distances = celeb_identifier.identify(image)
        # print(labels, distances)
        
        image = draw_rectangles(image, face_boxes)
        image = draw_landmarks(image, face_landmarks)
        for face_box, label, distance in zip(face_boxes, labels, distances):
            text = '{}: {:.3f}'.format(label[0], distance[0])
            position = (int(face_box[0] + 2), int(face_box[1] - 20))
            image = draw_text(image, text, position)
            
        cv2.imshow('image', image)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
