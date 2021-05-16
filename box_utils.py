import tensorflow as tf


def convert_to_square(bboxes):
    """Convert bounding boxes to a square form.

    Parameters:
        bboxes: float tensor of shape [n, 4]

    Returns:
        float tensor of shape [n, 4]
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1
    w = x2 - x1
    max_side = tf.maximum(h, w)

    '''
    choose one size , recoordinate coord to the square
    '''
    dx1 = x1 + w * 0.5 - max_side * 0.5
    dy1 = y1 + h * 0.5 - max_side * 0.5
    dx2 = dx1 + max_side
    dy2 = dy1 + max_side

    #box :  [dx1,dy1,dx1+max_sie,dy1+max_size]--> box: h= w = max_size
    return tf.stack([
        tf.math.round(dx1),
        tf.math.round(dy1),
        tf.math.round(dx2),
        tf.math.round(dy2),
    ], 1)


def calibrate_box(bboxes, offsets):
    """Use offsets returned by a network to
    correct the bounding box coordinates.

    Parameters:
        bboxes: float tensor of shape [n, 4].
        offsets: float tensor of shape [n, 4].

    Returns:
        float tensor of shape [n, 4]
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1
    h = y2 - y1

    #(w,h,w,h)*delta+(x1,y1,x2,y2), each size:(n,)
    translation = tf.stack([w, h, w, h], 1) * offsets
    return bboxes + translation #e.g. x1_true = delta_x1*(x2-x1)+x1


def get_image_boxes(bboxes, img, height, width, num_boxes, size=24):
    """Cut out boxes from the image.

    Parameters:
        bboxes: float tensor of shape [n, 4]
        img: image tensor
        height: float, image height
        width: float, image width
        num_boxes: int, number of rows in bboxes
        size: int, size of cutouts

    Returns:
        float tensor of shape [n, size, size, 3]
    """
    x1 = tf.math.maximum(bboxes[:, 0], 0.0) / width
    y1 = tf.math.maximum(bboxes[:, 1], 0.0) / height
    x2 = tf.math.minimum(bboxes[:, 2], width) / width
    y2 = tf.math.minimum(bboxes[:, 3], height) / height
    boxes = tf.stack([y1, x1, y2, x2], 1) #see https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize
    img_boxes = tf.image.crop_and_resize(tf.expand_dims(img, 0), boxes,
                                         tf.zeros(num_boxes, dtype=tf.int32),
                                         (size, size))
    return img_boxes


def generate_bboxes(probs, offsets, scale, threshold):
    """Convert output of PNet to bouding boxes tensor.

    Parameters:
        probs: float tensor of shape [p, m, 2], output of PNet
        offsets: float tensor of shape [p, m, 4], output of PNet
        scale: float, scale of the input image
        threshold: float, confidence threshold

    Returns:
        float tensor of shape [n, 9]
    """
    stride = 2
    cell_size = 12

    probs = probs[:, :, 1]  #pick object class

    # indices of boxes where there is probably a face
    # inds: N x 2
    #[[h0,w0],...]
    inds = tf.where(probs > threshold) #boolean

    if inds.shape[0] == 0:
        return tf.zeros((0, 9)) #no object

    # offsets: N x 4
    offsets = tf.gather_nd(offsets, inds)

    # score: N x 1
    score = tf.expand_dims(tf.gather_nd(probs, inds), axis=1)

    inds = tf.cast(inds, tf.float32)  #boolean to float32

    # bounding_boxes: N x 9
    #h_grids:inds[:.0]   w_grids:inds[:,1]
    bounding_boxes = tf.concat([
        tf.expand_dims(tf.math.round((stride * inds[:, 1]) / scale), 1), #x1
        tf.expand_dims(tf.math.round((stride * inds[:, 0]) / scale), 1), #y1
        tf.expand_dims(tf.math.round((stride * inds[:, 1] + cell_size) / scale), 1), #x2
        tf.expand_dims(tf.math.round((stride * inds[:, 0] + cell_size) / scale), 1), #y2
        score, offsets
    ], axis=1)
    return bounding_boxes  #Nx9


def preprocess(img):
    img = (img - 127.5) * (1./127.5)
    return img
