# -*- coding:utf-8 -*-
import tensorflow as tf


def SegNet_model(input_shape=(512, 512, 3), classes=3):

    h = inputs = tf.keras.Input(input_shape)    # per_image_standliazation 을 시켜야함

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", name="conv1")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", name="conv2")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", name="conv3")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", name="conv4")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="conv5")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="conv6")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same", name="conv7")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)
    
    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv8")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv9")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv10")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv11")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv12")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same", name="conv13")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2, padding="same")(h)

    ######################################################################################################

    h = tf.keras.layers.UpSampling2D(size=(2,2))(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.UpSampling2D(size=(2,2))(h)
    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.UpSampling2D(size=(2,2))(h)
    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.UpSampling2D(size=(2,2))(h)
    h = tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.UpSampling2D(size=(2,2))(h)
    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same")(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)

    h = tf.keras.layers.Conv2D(filters=classes, kernel_size=1, padding="same")(h)

    return tf.keras.Model(inputs=inputs, outputs=h)
