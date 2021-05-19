import tensorflow as tf


#img data augmentation trick
def image_color_distort(inputs):
    inputs = tf.image.random_contrast(inputs, lower=0.5, upper=1.5)
    inputs = tf.image.random_brightness(inputs, max_delta=0.2)
    inputs = tf.image.random_hue(inputs,max_delta= 0.2)
    inputs = tf.image.random_saturation(inputs,lower = 0.5, upper= 1.5)
    return inputs



def red_tf(imgs,net_size):
    print(11)
    raw_image_dataset = tf.data.TFRecordDataset(imgs).shuffle(1000)

    image_feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/label': tf.io.FixedLenFeature([], tf.int64),
        'image/roi': tf.io.FixedLenFeature([4], tf.float32),
        #'image/roi': tf.io.FixedLenFeature([14], tf.float32) #14 is for onet data length
    }
    def _parse_image_function(example_proto):

      return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
    print(parsed_image_dataset)
    image_batch = []
    label_batch = []
    bbox_batch = []

    for idx,image_features in enumerate(parsed_image_dataset):
        print(idx)
        image_raw = tf.io.decode_raw(image_features['image/encoded'],tf.uint8)

        images = tf.reshape(image_raw, [net_size, net_size, 3])
        image = (tf.cast(images, tf.float32) - 127.5) / 128

        image = image_color_distort(image)
        image_batch.append(image)

        label = tf.cast(image_features['image/label'], tf.float32)
        label_batch.append(label)

        roi = tf.cast(image_features['image/roi'], tf.float32)
        bbox_batch.append(roi)


    return image_batch,label_batch,bbox_batch
