from mtcnn import MTCNN
import cv2



def draw_faces(img, bboxes, landmarks):
    for box in bboxes:
        img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])), (255, 0, 0), 2)
    return img


pnet_weights_path='./train/Weights/pnet_weights_file/pnet_weights'

mtcnn=MTCNN(pnet_path=pnet_weights_path)

img=cv2.imread('./test_imgs/test.jpg')
boxes=mtcnn(img)

img = draw_faces(img, boxes, None)
cv2.imwrite('./test_imgs/result_rnet.jpg', img)
