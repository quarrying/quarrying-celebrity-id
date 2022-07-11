import time

import cv2
import khandy
import numpy as np

from celebid import CelebrityIdentifier


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
    for i, landmark in enumerate(landmarks):
        for x, y in landmark:
            cv2.circle(image, (int(x), int(y)), 1, (0,255,0), -1, 8)
    return image
    
    
if __name__ == '__main__':
    celeb_identifier = CelebrityIdentifier(size_thresh=40)
    filenames = khandy.get_all_filenames('images')
    filenames += khandy.get_all_filenames(r'G:\Human\Celebrity\_gallery\astronaut_chinese')
    for k, filename in enumerate(filenames):
        print('[{}/{}] {}'.format(k+1, len(filenames), filename))
        image = khandy.imread_cv(filename, 1)
        if (image is None) or (image.dtype != np.uint8):
            print('Image file corrupted!')
            continue
        start_time = time.time()
        face_boxes, face_landmarks, labels, distances = celeb_identifier.identify(image)
        print('Elapsed: {:.3f}s'.format(time.time() - start_time))

        if min(image.shape[:2]) > 1080:
            image = khandy.resize_image_short(image, 1080)
        image = draw_rectangles(image, face_boxes)
        image = draw_landmarks(image, face_landmarks)
        for face_box, label, distance in zip(face_boxes, labels, distances):
            text = '{}: {:.3f}'.format(label[0], distance[0])
            position = (int(face_box[0] + 2), int(face_box[1] - 20))
            image = khandy.draw_text(image, text, position, font='simsun.ttc', font_size=15)

        cv2.imshow('image', image)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()
