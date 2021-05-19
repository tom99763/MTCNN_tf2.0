from mtcnn import MTCNN
import cv2


def draw_faces(img, bboxes, landmarks, scores):

    for box, landmark, score in zip(bboxes, landmarks, scores):
        img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])), (255, 0, 0), 2)
        for i in range(0,9,2):
            x = int(landmark[i])
            y = int(landmark[i + 1])
            img = cv2.circle(img, (x, y), 2, (0, 255, 0))

        img = cv2.putText(img, '{:.2f}'.format(score), (int(box[0]), int(box[1])),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
    return img


pnet_weights_path='./train/Weights/pnet_weights_file/pnet_weights'
rnet_weights_path='./train/Weights/rnet_weights_file/rnet_weights'
onet_weights_path='./train/Weights/onet_weights_file/onet_weights'

mtcnn=MTCNN(pnet_path=pnet_weights_path,rnet_path=rnet_weights_path,onet_path=onet_weights_path)

name = 'test'

img=cv2.imread(f'./test/{name}.jpg')


'''
boxes=mtcnn.p_step(img)
img_res = draw_faces(img, boxes, None, None)
cv2.imwrite(f'./test/{name}_result_pnet2.jpg', img_res)
'''


boxes,landmarks,scores=mtcnn(img)
img_res = draw_faces(img, boxes, landmarks, scores)
cv2.imwrite(f'./test/{name}_result.jpg', img_res)
