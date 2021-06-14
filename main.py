from mtcnn import MTCNN
from align_faces import align_faces
from crop_faces import clean_faces
from crop_faces import crop_faces
from crop_faces import crop_faces_video


if __name__ == '__main__':
    detector = MTCNN()

    crop_faces('F:/_Data/Celebrity/_gallery/star_chinese_W nonface', detector)
    