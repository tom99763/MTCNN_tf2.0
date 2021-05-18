import tensorflow as tf
import tensorflow.keras as keras
from read_tfrecord import *
from loss_function import cls_ohem,bbox_ohem,landmark_ohem
from model import Onet
from tqdm import tqdm

data_path = "../data/preprocessing/48/train_ONet_landmark.tfrecord_shuffle"
batch_size = 64
lr=1e-4
#model_save='./Weights/rnet_weights_file/rnet_weights'
model_save=None
epochs = 20

def load_ds():

    size = 48
    print(1)
    images,labels,boxes = red_tf(data_path,size) #preprocessing normalize....

    print(3)
    ima = tf.data.Dataset.from_tensor_slices(images)
    print(4)
    lab = tf.data.Dataset.from_tensor_slices(labels)
    print(5)
    roi = tf.data.Dataset.from_tensor_slices(boxes)
    print(6)
    train_data = tf.data.Dataset.zip((ima, lab, roi)).shuffle(120000).batch(batch_size)
    print(7)
    train_data = list(train_data.as_numpy_iterator())
    return train_data


def train(eopch):
    model = Onet()
    if model_save:
        print('load_weights')
        model.load_weights(model_save)

    optimizer = keras.optimizers.Adam(learning_rate=lr)

    ds_train=load_ds()

    for epoch in tqdm(range(eopch)):

        for i,(img,lab,boxes) in enumerate(ds_train):

            with tf.GradientTape() as tape:
                cls_prob, bbox_pred,landmark_pred = model(img)
                cls_loss = cls_ohem(cls_prob, lab)
                bbox_loss = bbox_ohem(bbox_pred, boxes[:,4],lab)
                landmark_loss = landmark_ohem(landmark_pred, boxes[4:], lab)
                total_loss_value = cls_loss + 0.5 * bbox_loss+ landmark_loss

            grads = tape.gradient(total_loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if i % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (i, float(total_loss_value)))
                print('Seen so far: %s samples' % ((i + 1) * 6))

    model.save_weights('./Weights/rnet_weights_file/rnet_weights')

train(epochs)