import tensorflow as tf


def Pnet():
    input = tf.keras.Input(shape=[None, None, 3])
    x = tf.keras.layers.Conv2D(10, (3, 3), name='conv1',kernel_regularizer=keras.regularizers.l2(0.0005))(input)
    x = tf.keras.layers.PReLU(tf.constant_initializer(0.25),shared_axes=[1, 2], name='PReLU1')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(16, (3, 3),name='conv2',kernel_regularizer=keras.regularizers.l2(0.0005))(x)
    x = tf.keras.layers.PReLU(tf.constant_initializer(0.25),shared_axes=[1, 2], name='PReLU2')(x)
    x = tf.keras.layers.Conv2D(32, (3, 3),name='conv3',kernel_regularizer=keras.regularizers.l2(0.0005))(x)
    x = tf.keras.layers.PReLU(tf.constant_initializer(0.25),shared_axes=[1, 2], name='PReLU3')(x)
    classifier = tf.keras.layers.Conv2D(2, (1, 1), activation='softmax',name='conv4-1')(x)
    bbox_regress = tf.keras.layers.Conv2D(4, (1, 1), name='conv4-2')(x)
    model = tf.keras.models.Model([input], [classifier, bbox_regress])
    return model


def Rnet():
    input = tf.keras.Input(shape=[24, 24, 3])
    x = tf.keras.layers.Conv2D(28, (3, 3),strides=1,padding='valid',name='conv1')(input)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2],name='prelu1')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3,strides=2,padding='same')(x)
    x = tf.keras.layers.Conv2D(48, (3, 3),strides=1,padding='valid',name='conv2')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2],name='prelu2')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=3,strides=2)(x)
    x = tf.keras.layers.Conv2D(64, (2, 2),strides=1,padding='valid',name='conv3')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2],name='prelu3')(x)
    x = tf.keras.layers.Permute((3, 2, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, name='conv4')(x)
    x = tf.keras.layers.PReLU(name='prelu4')(x)
    classifier = tf.keras.layers.Dense(2,activation='softmax',name='conv5-1')(x)
    bbox_regress = tf.keras.layers.Dense(4, name='conv5-2')(x)
    model = tf.keras.models.Model([input], [classifier, bbox_regress])
    return model


def Onet():
    input = tf.keras.layers.Input(shape=[48, 48, 3])
    x = tf.keras.layers.Conv2D(32, (3, 3),strides=1,padding='valid',name='conv1')(input)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2],name='prelu1')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3,strides=2,padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3),strides=1,padding='valid',name='conv2')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2],name='prelu2')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3,strides=2)(x)
    x = tf.keras.layers.Conv2D(64, (3, 3),strides=1,padding='valid',name='conv3')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2],name='prelu3')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(128, (2, 2),strides=1,padding='valid',name='conv4')(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2],name='prelu4')(x)
    x = tf.keras.layers.Permute((3, 2, 1))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, name='conv5')(x)
    x = tf.keras.layers.PReLU(name='prelu5')(x)

    classifier = tf.keras.layers.Dense(2,activation='softmax',name='conv6-1')(x)
    bbox_regress = tf.keras.layers.Dense(4, name='conv6-2')(x)
    landmark_regress = tf.keras.layers.Dense(10, name='conv6-3')(x)
    model = tf.keras.models.Model([input], [classifier, bbox_regress,landmark_regress])
    return model

