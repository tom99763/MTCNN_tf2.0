import os
import random
import sys
import tensorflow as tf
import cv2

'''
tf.train.Feature:only eat bytelist、floatlist、int64list
'''


def _int64_feature(value):
    #for img
    #return int64list from a bool/enum/int/uint
    """Wrapper for insert int64 feature into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    #for box coord
    #return floatlist from a float/double
    """Wrapper for insert float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    #for txt
    #return bytelist from a string / byte
    """Wrapper for insert bytes features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value)) #010100101+壓縮



def _process_image_withoutcoder(filename):

    image = cv2.imread('./12/'+filename.replace('.jpg.jpg','.jpg'))

    # transform data into string format
    image_data = image.tobytes()
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3
    # return string data and initial height and width of the image
    return image_data, height, width


def _convert_to_example_simple(image_example, image_buffer):
    '''e.g.
    def serialize_example(feature0, feature1, feature2, feature3):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {'feature0': _int64_feature(feature0),'feature1': _int64_feature(feature1),
    'feature2': _bytes_feature(feature2),'feature3': _float_feature(feature3),}
    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
    '''
    """
        covert to tfrecord file
    Parameter
    ------------
        image_example: dict, an image example
        image_buffer: string, JPEG encoding of RGB image
    Return
    -----------
        Example proto
    """
    class_label = image_example['label']

    bbox = image_example['bbox']
    roi = [bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']]
    # landmark = [bbox['xlefteye'],bbox['ylefteye'],bbox['xrighteye'],bbox['yrighteye'],bbox['xnose'],bbox['ynose'],
    #             bbox['xleftmouth'],bbox['yleftmouth'],bbox['xrightmouth'],bbox['yrightmouth']]

    # 壓縮 {name:壓縮的資料,...}
    #tf.train.Example:封装一筆data
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': _bytes_feature(image_buffer),
        'image/label': _int64_feature(class_label),
        'image/roi': _float_feature(roi),
        # 'image/landmark': _float_feature(landmark)
    }))

    return example



def _add_to_tfrecord(filename, image_example, tfrecord_writer):
    # print('---', filename)


    image_data, height, width = _process_image_withoutcoder(filename)
    example = _convert_to_example_simple(image_example, image_data)
    tfrecord_writer.write(example.SerializeToString())


def _get_output_filename(output_dir,net):

    return '%s/train_%s_landmark.tfrecord' % (output_dir,net)


def run(dataset_dir,net,output_dir,shuffling=False):

    tf_filename = _get_output_filename(output_dir,net)

    if tf.io.gfile.exists(tf_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return


    dataset = get_dataset(dataset_dir)
    print(dataset)

    if shuffling:
        tf_filename = tf_filename + '_shuffle'

        random.shuffle(dataset)


    print('lala')
    with tf.io.TFRecordWriter(tf_filename) as tfrecord_writer:
        for i, image_example in enumerate(dataset):
            if (i + 1) % 1 == 0:
                sys.stdout.write('\r>> %d/%d images has been converted' % (
                    i + 1, len(dataset)))
            sys.stdout.flush()
            filename = image_example['filename']
            _add_to_tfrecord(filename, image_example, tfrecord_writer)



    print('\nFinished converting the MTCNN dataset!')


def get_dataset(dir):

    item = 'label-train%s.txt'%(dir)

    dataset_dir = os.path.join(dir, item)
    # print(dataset_dir)
    imagelist = open(dataset_dir, 'r')

    dataset = []  
    for line in imagelist.readlines():
        info = line.strip().split(' ')
        data_example = dict() 
        bbox = dict()
        data_example['filename'] = info[0]  # filename=info[0]
        # print(data_example['filename'])
        data_example['label'] = int(info[1])
        bbox['xmin'] = 0
        bbox['ymin'] = 0
        bbox['xmax'] = 0
        bbox['ymax'] = 0
        # bbox['xlefteye'] = 0
        # bbox['ylefteye'] = 0
        # bbox['xrighteye'] = 0
        # bbox['yrighteye'] = 0
        # bbox['xnose'] = 0
        # bbox['ynose'] = 0
        # bbox['xleftmouth'] = 0
        # bbox['yleftmouth'] = 0
        # bbox['xrightmouth'] = 0
        # bbox['yrightmouth'] = 0
        if len(info) == 6:
            bbox['xmin'] = float(info[2])
            bbox['ymin'] = float(info[3])
            bbox['xmax'] = float(info[4])
            bbox['ymax'] = float(info[5])
        # if len(info) == 12:
        #     bbox['xlefteye'] = float(info[2])
        #     bbox['ylefteye'] = float(info[3])
        #     bbox['xrighteye'] = float(info[4])
        #     bbox['yrighteye'] = float(info[5])
        #     bbox['xnose'] = float(info[6])
        #     bbox['ynose'] = float(info[7])
        #     bbox['xleftmouth'] = float(info[8])
        #     bbox['yleftmouth'] = float(info[9])
        #     bbox['xrightmouth'] = float(info[10])
        #     bbox['yrightmouth'] = float(info[11])

        data_example['bbox'] = bbox
        dataset.append(data_example)

    return dataset


if __name__ == '__main__':
    dir = '12'
    net = 'PNet'
    output_directory = '12'
    '''
    dir = '24'
    net = 'RNet'
    output_directory = '24'
    
    
    dir = '48'
    net = 'ONet'
    output_directory = '48'
    '''
    run(dir,net,output_directory,shuffling=True)
