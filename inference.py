from mtcnn import MTCNN
import cv2


def draw_faces(img, bboxes, landmarks, scores):
    if landmarks==None:
        for box in bboxes:
            img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])), (0, 255, 0), 2)
    else:
        for box, landmark, score in zip(bboxes, landmarks, scores):
            img = cv2.rectangle(img, (int(box[0]), int(box[1])),
                                (int(box[2]), int(box[3])), (0, 255, 0), 2)
            for i in range(0, 9, 2):
                x = int(landmark[i])
                y = int(landmark[i + 1])
                img = cv2.circle(img, (x, y), 3, (255, 0, 0))

            img = cv2.putText(img, '{:.2f}'.format(score), (int(box[0]), int(box[1])),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
    return img


def video_show(video=None, output=None,mtcnn=None):
    if video is None:
        vid = cv2.VideoCapture(0)
    else:
        vid = cv2.VideoCapture(video)

    out = None
    if output is not None:
        fps = vid.get(cv2.CAP_PROP_FPS)
        _, img = vid.read()
        h, w, _ = img.shape
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output, fourcc, fps, (w, h))

    while True:
        _, img = vid.read()
        if img is None:
            break
        try:
            bboxes, landmarks, scores = mtcnn(img)
        except:
            out.write(img)
            continue
        img = draw_faces(img, bboxes, landmarks, scores)

        cv2.imshow('show', img)
        if out is not None:
            out.write(img)
        if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
            break
    cv2.destroyAllWindows()



pnet_weights_path='./train/Weights/pnet_weights_file/pnet_weights'
rnet_weights_path='./train/Weights/rnet_weights_file/rnet_weights'
onet_weights_path='./train/Weights/onet_weights_file/onet_weights'
mtcnn=MTCNN(pnet_path=pnet_weights_path,rnet_path=rnet_weights_path,onet_path=onet_weights_path)

video_path='./test/andrew_wiggins.mp4'
video_show(video_path,'test/andrew_wiggins_res.mp4',mtcnn)




'''
name = 'splash_brothers'
img=cv2.imread(f'./test/{name}.jpg')
'''

'''
boxes=mtcnn.p_step(img)
img_res = draw_faces(img, boxes, None, None)
cv2.imwrite(f'./test/{name}_result_pnet.jpg', img_res)
'''

'''
boxes=mtcnn.p_step(img)
boxes=mtcnn.r_step(img,boxes)
img_res = draw_faces(img, boxes, None, None)
cv2.imwrite(f'./test/{name}_result_rnet.jpg', img_res)
'''

'''
boxes,landmarks,scores=mtcnn(img)
img_res = draw_faces(img, boxes, landmarks, scores)
cv2.imwrite(f'./test/{name}_result_output.jpg', img_res)
'''

