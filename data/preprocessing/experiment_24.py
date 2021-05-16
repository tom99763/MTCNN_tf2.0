import os
import cv2
import numpy as np
import numpy.random as npr

from utils import IoU

stdsize = 24

anno_file = '../face_detection/WIDERFACE/wider_face_split/wider_face_train.txt'
im_dir = '../face_detection/WIDERFACE/WIDER_train/WIDER_train/images/'
pos_save_dir = str(stdsize) + "/positive"
part_save_dir = str(stdsize) + "/part"
neg_save_dir = str(stdsize) + '/negative'
save_dir = "24"


if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(pos_save_dir):
    os.mkdir(pos_save_dir)
if not os.path.exists(part_save_dir):
    os.mkdir(part_save_dir)
if not os.path.exists(neg_save_dir):
    os.mkdir(neg_save_dir)


f1 = open(os.path.join(save_dir, 'pos_24.txt'), 'w')
f2 = open(os.path.join(save_dir, 'neg_24.txt'), 'w')
f3 = open(os.path.join(save_dir, 'part_24.txt'), 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print("%d pics in total" % num)
p_idx = 0
n_idx = 0
d_idx = 0
idx = 0
box_idx = 0
for annotation in annotations:
    annotation = annotation.strip().split(' ')



    im_path = annotation[0]

    bbox = list(map(float, annotation[1:]))
    # gt
    boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)

    img = cv2.imread(os.path.join(im_dir, im_path + '.jpg'))
    idx += 1


    try:
        height, width, channel = img.shape
    except:
        continue

    neg_num = 0

    count=0
    while neg_num < 50:
        count+=1
        if count>50000:
            break

        size = npr.randint(24, min(width, height) / 2)
        # top_left coordinate
        nx = npr.randint(0, width - size)
        ny = npr.randint(0, height - size)

        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = IoU(crop_box, boxes)



        cropped_im = img[ny: ny + size, nx: nx + size, :]

        resized_im = cv2.resize(cropped_im, (24, 24),
                                interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:

            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
            f2.write("negative/%s.jpg" % n_idx + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1


    for box in boxes:

        x1, y1, x2, y2 = box

        w = x2 - x1 + 1

        h = y2 - y1 + 1



        if max(w, h) < 20 or x1 < 0 or y1 < 0:
            continue


        for i in range(5):
            # size of the image to be cropped
            size = npr.randint(24, min(width, height) / 2)

            delta_x = npr.randint(max(-size, -x1), w)
            delta_y = npr.randint(max(-size, -y1), h)
            # max here not really necessary
            nx1 = int(max(0, x1 + delta_x))
            ny1 = int(max(0, y1 + delta_y))


            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]

            resized_im = cv2.resize(cropped_im, (24, 24), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:

                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write("negative/%s.jpg" % n_idx + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1


        for i in range(20):

            size = npr.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))


            if w < 5:
                print(w)
                continue


            delta_x = npr.randint(-w * 0.2, w * 0.2)
            delta_y = npr.randint(-h * 0.2, h * 0.2)



            nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
            ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
            nx2 = nx1 + size
            ny2 = ny1 + size


            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])


            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)

            cropped_im = img[ny1: ny2, nx1: nx2, :]

            resized_im = cv2.resize(cropped_im, (24, 24), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            iou = IoU(crop_box, box_)
            if iou >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                f1.write("positive/%s.jpg" % p_idx + ' 1 %.2f %.2f %.2f %.2f\n' % (
                offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif iou >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                f3.write("part/%s.jpg" % d_idx + ' -1 %.2f %.2f %.2f %.2f\n' % (
                offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1
        if idx % 100 == 0:
            print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))
f1.close()
f2.close()
f3.close()
