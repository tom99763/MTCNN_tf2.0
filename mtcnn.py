from train.model import Pnet,Rnet,Onet
import tensorflow as tf
from box_utils import calibrate_box, convert_to_square, get_image_boxes, generate_bboxes, preprocess



class MTCNN(object):
    def __init__(self, pnet_path=None, rnet_path=None, onet_path=None,
                 min_face_size=20.0,
                 thresholds= [0.5, 0.5, 0.5],
                 nms_thresholds=[0.6, 0.6, 0.6],
                 max_nms_output_num=300,
                 scale_factor=0.707):  #0.707 is empirical, no theory proof
        self.pnet = Pnet()
        if self.pnet_path:
            self.pnet.load_weights(pnet_path)
        #self.rnet = RNet(rnet_path)
        #self.onet = ONet(onet_path)
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.nms_thresholds = nms_thresholds
        self.max_nms_output_num = max_nms_output_num
        self.scale_cache = {}
        self.scale_factor=scale_facotr

    def __call__(self, img):
        bboxes = self.p_step(img)

        if bboxes.shape[0] == 0:
            return []
        bboxes = self.r_step(img,bboxes)

        if bboxes.shape[0] == 0:
            return []
        
        return bboxes


    def build_scale_pyrammid(self, height, width):
        min_length = min(height, width)

        if min_length in self.scale_cache:
            return self.scale_cache[min_length]

        min_detection_size = 12
        scales = []

        m = min_detection_size / self.min_face_size
        min_length *= m
        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * self.scale_factor**factor_count)
            min_length *= self.scale_factor
            factor_count += 1

        self.scale_cache[min_length] = scales
        return scales



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
        boxes = generate_bboxes(probs[0], offsets[0], scale, self.thresholds[0])
        if len(boxes) == 0:
            return boxes
        keep = tf.image.non_max_suppression(boxes[:, 0:4], boxes[:, 4], self.max_nms_output_num,
                                            iou_threshold=0.5)

        boxes = tf.gather(boxes, keep)
        return boxes


    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, 9), dtype=tf.float32)])
    def p_step_box_alignment(self, boxes):
        bboxes, scores, offsets = boxes[:, :4], boxes[:, 4], boxes[:, 5:]
        bboxes = calibrate_box(bboxes, offsets)
        bboxes = convert_to_square(bboxes)

        nms_idx = tf.image.non_max_suppression(bboxes, scores, self.max_nms_output_num,
                                            iou_threshold=self.nms_thresholds[0])
        bboxes = tf.gather(bboxes, nms_idx)
        return bboxes


    def p_step(self, img):
        img=preprocess(img)
        height, width, _ = img.shape

        img = tf.convert_to_tensor(img, tf.float32)
        scales = self.build_scale_pyrammid(height, width)

        boxes = []
        for s in scales:
            boxes.append(self.scale_search(img, height, width, s))
        boxes = tf.concat(boxes, 0)

        if boxes.shape[0] == 0:
            return []

        return self.p_step_box_alignment(boxes)
    
    def r_step(self, img, bboxes):
        img=preprocess(img)
        height,width,_=img.shape
        num_boxes=bboxes.shape[0]
        img_boxes = get_image_boxes(bboxes, img, height, width, num_boxes, size=24)
        probs, offsets = self.rnet(img_boxes)
        valid_idx =tf.argmax(probs,axis=-1)==1
        
        bboxes = tf.boolean_mask(bboxes,valid_idx)
        offsets = tf.boolean_mask(offsets,valid_idx)
        scores = tf.boolean_mask(probs[:,1],valid_idx)
        bboxes = calibrate_box(bboxes, offsets)
        bboxes = convert_to_square(bboxes)
        nms_idx = tf.image.non_max_suppression(bboxes, scores,
                                            self.max_output_size, self.nms_thresholds[1])
        bboxes = tf.gather(bboxes, nms_idx)
        return bboxes




'''
mtcnn=MTCNN()
x=tf.random.normal((32,32,3))
mtcnn(x)
'''
