from train.model import Pnet, Rnet, Onet
import tensorflow as tf
from box_utils import calibrate_box, convert_to_square, get_image_boxes, generate_bboxes, preprocess


class MTCNN(object):
    def __init__(self, pnet_path=None, rnet_path=None, onet_path=None,
                 min_face_size=20.0,
                 thresholds=[0.5, 0.5, 0.5],
                 nms_thresholds=[0.6, 0.6, 0.00001],
                 max_nms_output_num=300,
                 scale_factor=0.707):  # 0.707 is empirical, no theory proof
        self.pnet = Pnet()
        if pnet_path:
            self.pnet.load_weights(pnet_path)
        self.rnet = Rnet()
        if rnet_path:
            self.rnet.load_weights(rnet_path)
        self.onet = Onet()
        if onet_path:
             self.onet.load_weights(onet_path)
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.nms_thresholds = nms_thresholds
        self.max_nms_output_num = max_nms_output_num
        self.scale_factor = scale_factor

    def __call__(self, img):
        bboxes = self.p_step(img)

        if len(bboxes) == 0:
            return [],[],[]
        bboxes = self.r_step(img, bboxes)

        if len(bboxes) == 0:
            return [],[],[]

        bboxes,landmarks,scores = self.o_step(img,bboxes)


        if len(bboxes) == 0:
            return [],[],[]

        return bboxes ,landmarks, scores

    def build_scale_pyramid(self, img, height, width):
        infos = []
        min_length = min(height, width)
        min_detection_size = 12
        m = min_detection_size / self.min_face_size
        min_length *= m
        factor_count = 0
        while min_length > min_detection_size:
            scale = m * self.scale_factor ** factor_count
            min_length *= self.scale_factor
            factor_count += 1
            infos.append(self.scale_search(img, height, width, scale))
        return infos

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.float32)])
    def scale_search(self, img, height, width, scale):
        hs = tf.math.ceil(height * scale)
        ws = tf.math.ceil(width * scale)
        img_in = tf.image.resize(img, (hs, ws))
        img_in = tf.expand_dims(img_in, 0)

        probs, offsets = self.pnet(img_in)
        info = generate_bboxes(probs[0], offsets[0], scale, self.thresholds[0])
        if info.shape[0] == 0:
            return info
        nms_idx = tf.image.non_max_suppression(info[:, 0:4], info[:, 4], self.max_nms_output_num,
                                               iou_threshold=0.5)
        info = tf.gather(info, nms_idx)
        return info

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                         tf.TensorSpec(shape=(None, 4), dtype=tf.float32)])
    def bbox_alignment(self, bboxes, offsets):
        bboxes = calibrate_box(bboxes, offsets)
        bboxes = convert_to_square(bboxes)
        return bboxes

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                                  tf.TensorSpec(shape=(None, 10), dtype=tf.float32)])
    def landmark_alignment(self,bboxes,landmarks):
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w = x2 - x1
        h = y2 - y1

        landmarks = tf.stack([landmarks[:, 0] * w + x1,
                              landmarks[:, 1] * h + y1,
                              landmarks[:, 2] * w + x1,
                              landmarks[:, 3] * h + y1,
                              landmarks[:, 4] * w + x1,
                              landmarks[:, 5] * h + y1,
                              landmarks[:, 6] * w + x1,
                              landmarks[:, 7] * h + y1,
                              landmarks[:, 8] * w + x1,
                              landmarks[:, 9] * h + y1])

        landmarks = tf.transpose(landmarks)

        return landmarks


    def p_step(self, img):
        img = preprocess(img)
        height, width, _ = img.shape
        img = tf.convert_to_tensor(img, tf.float32)

        infos = self.build_scale_pyramid(img, height, width)
        infos = tf.concat(infos, 0)

        if infos.shape[0] == 0:
            return []

        bboxes, scores, offsets = infos[:, :4], infos[:, 4], infos[:, 5:]
        bboxes = self.bbox_alignment(bboxes, offsets)

        nms_idx = tf.image.non_max_suppression(bboxes, scores, self.max_nms_output_num,
                                               iou_threshold=self.nms_thresholds[0])
        bboxes = tf.gather(bboxes, nms_idx)

        return bboxes

    def r_step(self, img, bboxes):
        img = preprocess(img)
        height, width, _ = img.shape
        num_boxes = bboxes.shape[0]
        img_boxes = get_image_boxes(bboxes, img, height, width, num_boxes, size=24)
        probs, offsets = self.rnet(img_boxes)
        valid_idx = tf.argmax(probs, axis=-1) == 1

        bboxes = tf.boolean_mask(bboxes, valid_idx)

        if bboxes.shape[0] == 0:
            return []

        offsets = tf.boolean_mask(offsets, valid_idx)
        scores = tf.boolean_mask(probs[:, 1], valid_idx)

        bboxes = self.bbox_alignment(bboxes, offsets)

        nms_idx = tf.image.non_max_suppression(bboxes, scores, self.max_nms_output_num,
                                               iou_threshold=self.nms_thresholds[1])
        bboxes = tf.gather(bboxes, nms_idx)

        return bboxes

    def o_step(self,img,bboxes):
        img = preprocess(img)
        height, width, _ = img.shape
        num_boxes = bboxes.shape[0]
        img_boxes = get_image_boxes(bboxes, img, height, width, num_boxes, size=48)
        probs, offsets,landmarks = self.onet(img_boxes)
        valid_idx =probs[:, 1] > self.thresholds[-1]

        bboxes = tf.boolean_mask(bboxes, valid_idx)

        if bboxes.shape[0] == 0:
            return [],[],[]

        offsets = tf.boolean_mask(offsets, valid_idx)
        scores = tf.boolean_mask(probs[:, 1], valid_idx)
        landmarks = tf.boolean_mask(landmarks, valid_idx)

        landmarks = self.landmark_alignment(bboxes, landmarks)
        bboxes = calibrate_box(bboxes, offsets)
        nms_idx = tf.image.non_max_suppression(bboxes, scores, self.max_nms_output_num,
                                               iou_threshold=self.nms_thresholds[2])
        bboxes = tf.gather(bboxes, nms_idx)
        landmarks=tf.gather(landmarks,nms_idx)
        scores=tf.gather(scores,nms_idx)


        return bboxes,landmarks,scores
