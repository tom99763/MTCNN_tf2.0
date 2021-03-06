import numpy as np


def IoU(pr_box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """


    box_area = (pr_box[2] - pr_box[0] + 1) * (pr_box[3] - pr_box[1] + 1)


    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)


    xx1 = np.maximum(pr_box[0], boxes[:, 0])
    # print("xx1",xx1)
    yy1 = np.maximum(pr_box[1], boxes[:, 1])
    # print("yy1",yy1)
    xx2 = np.minimum(pr_box[2], boxes[:, 2])
    # print("xx2",xx2)
    yy2 = np.minimum(pr_box[3], boxes[:, 3])
    # print("yy2",yy2)
    # compute the width and height of the bounding box
    # print("xx2-xx1",(xx2-xx1))
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)
    # inter_area = (xx1 - xx2 + 1) * (yy1 - yy2 + 1)
    # w = np.max(xx1,yy1)

    inter = w * h
    # print("inter",inter_area)
    ovr = inter / (box_area + area - inter)
    return ovr
