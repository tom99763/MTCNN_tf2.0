
import tensorflow as tf


def cls_ohem(cls_prob, label):

    '''
    Args:
        cls_prob: (batch,2)
        label:  (batch,)

    Returns:
    '''

    zeros = tf.zeros_like(label, dtype=tf.float32)

    label_filter_invalid = tf.where(tf.math.less(label,[0]),zeros,label)


    num_cls_prob = tf.size(cls_prob)  #batch*2

    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])  #(batch*2,1) [clf0_0,clf1_0,clf0_1,clf1_1,....]

    label_int = tf.cast(label_filter_invalid,dtype=tf.int32)

    num_row = tf.cast(cls_prob.get_shape()[0],dtype=tf.int32)  #batch scaler


    row = tf.range(num_row)*2   #[0 2 4 ..... batch]

    indices_ = row + label_int

    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))



    loss = -tf.math.log(label_prob+1e-10)  #Egt~G(x)[-ln f(x)]


    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob, dtype=tf.float32)


    #num data which is 0 or 1 , skip -1 to compute loss
    valid_inds = tf.where(label < zeros,zeros,ones)

    num_valid = tf.reduce_sum(valid_inds)

    keep_num = tf.cast(num_valid*0.7,dtype=tf.int32)

    loss = loss * num_valid #mask -1 labeled data

    loss,_ = tf.math.top_k(loss, k=keep_num)  #pick top k high loss

    return tf.math.reduce_mean(loss)



def bbox_ohem(bbox_pred,bbox_target,label):

    zeros_index = tf.zeros_like(label,dtype=tf.float32)
    ones_index = tf.ones_like(label,dtype=tf.float32)


    valid_inds = tf.where(tf.math.equal(tf.math.abs(label),1),ones_index,zeros_index)


    square_error = tf.math.square(bbox_pred - bbox_target)  #16-1-16-14
    square_error = tf.math.reduce_sum(square_error,axis=1)  #16*16*4



    num_valid = tf.math.reduce_sum(valid_inds)
    keep_num = tf.cast(num_valid,dtype=tf.int32)



    square_error = square_error * num_valid


    _,k_index = tf.math.top_k(square_error,k=keep_num)

    square_error = tf.gather(square_error, k_index)

    return tf.reduce_mean(square_error)



def landmark_ohem(landmark_pred,landmark_target,label):

    ones = tf.ones_like(label,dtype=tf.float32)
    zeros = tf.zeros_like(label,dtype=tf.float32)


    valid_inds = tf.where(tf.equal(label,-2),ones,zeros)


    square_error = tf.square(landmark_pred-landmark_target)
    square_error = tf.reduce_sum(square_error,axis=1)


    num_valid = tf.math.reduce_sum(valid_inds) # 0
    keep_num = tf.cast(num_valid, dtype=tf.int32) # 0


    square_error = square_error*valid_inds
    square_error, k_index = tf.nn.top_k(square_error, k=keep_num)


    return tf.math.reduce_mean(square_error)




