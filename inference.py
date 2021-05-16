from mtcnn import MTCNN
import cv2



def draw_faces(img, bboxes, landmarks, scores):
    for box in bboxes:
        img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])), (255, 0, 0), 2)
    return img


pnet_weights_path='./train/Weights/pnet_weights_file/pnet_weights'
rnet_weights_path='./train/Weights/rnet_weights_file/rnet_weights'

mtcnn=MTCNN(pnet_path=pnet_weights_path,rnet_path=rnet_weights_path)

img=cv2.imread('test.jpg')



'''
boxes=mtcnn.p_step(img)
img_res = draw_faces(img, boxes, None, None)
cv2.imwrite('result_pnet.jpg', img_res)
'''


boxes=mtcnn(img)
img_res = draw_faces(img, boxes, None, None)
cv2.imwrite('result.jpg', img_res)
