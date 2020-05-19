import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.utils import plot_model
from tensorflow.keras.regularizers import l2


def vgg16():
    # params
    weight_decay = 0.0005

    input = Input(shape=[32, 32, 3])
    # block1
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(input)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool2D(pool_size=2, strides=1, padding='same')(x)
    # block2
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    # block3
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool2D(pool_size=2, strides=1, padding='same')(x)
    # block4
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    # block5
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = MaxPool2D(pool_size=2, strides=1, padding='same')(x)
    # dense
    x = Flatten()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    output = Dense(10, activation='softmax')(x)

    vgg16 =  keras.models.Model(inputs=[input],
                                outputs=[output])

    return vgg16


if __name__ == "__main__":
    model = vgg16()
    model.summary()

    depth = 16
    plot_model(model, to_file='vgg{}.png'.format(depth))