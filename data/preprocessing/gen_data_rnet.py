from utils import IoU
import numpy as np
import cv2
import os,sys
from mtcnn import MTCNN



stdsize = 24
anno_file = '../face_detection/WIDERFACE/wider_face_split/wider_face_train.txt'
im_dir = '../face_detection/WIDERFACE/WIDER_train/WIDER_train/images/'
pos_save_dir = str(stdsize) + "/positive"
part_save_dir = str(stdsize) + "/part"
neg_save_dir = str(stdsize) + '/negative'
pnet_weights = '../../train/Weights/pnet_weights_file/pnet_weights'
save_dir = "24"
model = MTCNN(pnet_path=pnet_weights)



def mkr(dr):
    if not os.path.exists(dr):
        os.mkdir(dr)

mkr(save_dir)
mkr(pos_save_dir)
mkr(part_save_dir)
mkr(neg_save_dir)


f1 = open(os.path.join(save_dir, 'pos_' + str(stdsize) + '.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_' + str(stdsize) + '.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_' + str(stdsize) + '.txt'), 'w')

with open(anno_file, 'r') as f:
    annotations = f.readlines()
    del annotations[0]
num = len(annotations)
print("%d pics in total" % num)
p_idx = 0 # positive
n_idx = 0 # negative
d_idx = 0 # dont care
idx = 0
box_idx = 0


size_i = 24



for annotation in annotations:
    annotation = annotation.strip().split(' ')

    im_path = annotation[0]

    bboxs = list(map(float, annotation[1:]))

    boxes = np.array(bboxs, dtype=np.float32).reshape(-1, 4)


    image = cv2.imread(im_dir+im_path+'.jpg')


    pre_bboxes= model.p_step(image)

    try:
        print("box num",pre_bboxes.shape[0])
    except:
        continue

    if len(pre_bboxes) == 0:
        continue


    dets = pre_bboxes.numpy()

    img = cv2.imread(im_dir+im_path+'.jpg')
    idx += 1

    height,width,channel = img.shape

    neg_num = 0
    for box in dets:

        x_left,y_top,x_right,y_bottom = box[0:4].astype(int)
        width = x_right - x_left + 1
        height = y_bottom - y_top + 1
        if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
            continue


        Iou = IoU(box,boxes)

        cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]

        resized_im = cv2.resize(cropped_im, (stdsize, stdsize),
                                interpolation=cv2.INTER_LINEAR)

        # if np.max(Iou) < 0.2 and n_idx < 3.0 * p_idx + 1:
        if np.max(Iou) < 0.3 and neg_num < 60:
            save_file = os.path.join(neg_save_dir,"%s.jpg"%n_idx)
            f2.write("negative/%s"% n_idx + " 0\n")
            cv2.imwrite(save_file,resized_im)
            n_idx += 1
            neg_num += 1

        else:
            idx_Iou = np.argmax(Iou)
            assigned_gt = boxes[idx_Iou]
            x1,y1,x2,y2 = assigned_gt

            offset_x1 = (x1 - x_left) / float(width)
            offset_y1 = (y1 - y_top) / float(height)
            offset_x2 = (x2 - x_right) / float(width)
            offset_y2 = (y2 - y_bottom) / float(height)
            if np.max(Iou) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                f1.write("positive/%s"%p_idx + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1

            elif np.max(Iou) >= 0.4 and d_idx < 1.0 * p_idx + 1:
                save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                f3.write("part/%s"%d_idx + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1

    print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))


f1.close()
f2.close()
f3.close()
