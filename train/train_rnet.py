import tensorflow as tf
import tensorflow.keras as keras
from read_tfrecord import red_tf
from loss_function import cls_ohem,bbox_ohem
from model import Rnet
from tqdm import tqdm
import os

data_path = "../data/preprocessing/24/train_RNet_landmark.tfrecord_shuffle"
batch_size = 64
lr=1e-4
model_save=None

def load_ds():

    size = 24
    print(1)
    images,labels,boxes = red_tf(data_path,size) #preprocessing normalize....

    print(3)
    ima = tf.data.Dataset.from_tensor_slices(images)
    print(4)
    lab = tf.data.Dataset.from_tensor_slices(labels)
    print(5)
    roi = tf.data.Dataset.from_tensor_slices(boxes)
    print(6)
    train_data = tf.data.Dataset.zip((ima, lab, roi)).shuffle(600000).batch(batch_size)
    print(7)
    train_data = list(train_data.as_numpy_iterator())
    return train_data


def train(eopch):
    model = Rnet()
    if model_save:
        model.load_weights(model_save)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    ds_train=load_ds()
    for epoch in tqdm(range(eopch)):

        for i,(img,lab,boxes) in enumerate(ds_train):
            img = image_color_distort(img)
            with tf.GradientTape() as tape:
                cls_prob, bbox_pred = model(img)
                cls_loss = cls_ohem(cls_prob, lab)

                bbox_loss = bbox_ohem(bbox_pred, boxes,lab)

                total_loss_value = cls_loss + 0.5 * bbox_loss
            grads = tape.gradient(total_loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            if i % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (i, float(total_loss_value)))
                print('Seen : %s samples' % ((i + 1) * 6))

    model.save_weights('./Weights/rnet_weights_file/rnet_weights')

train(20)

