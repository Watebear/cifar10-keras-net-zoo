import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dense, Dropout, Flatten, \
                                    BatchNormalization, Activation, AveragePooling2D,\
                                    MaxPool2D, concatenate, GlobalAveragePooling2D
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

def Inception_Block_v1(inputs, num_filters):
    '''
    b1 : 1x1x32 -> 3x3x64 -> 3x3x128
    b2 : 1x1x32 -> 3x3x128
    b3 : 1x1mp -> 1x1x12
    b4 : 1x1x128
    out : concat[b1, b2, b3, b4]
    '''
    x = inputs
    # branch1
    b1 = Conv2D(filters=int(num_filters/4), kernel_size=1, strides=1, padding='same', activation='relu')(x)
    b1 = Conv2D(filters=int(num_filters/2), kernel_size=3, strides=1, padding='same', activation='relu')(b1)
    b1 = Conv2D(filters=num_filters, kernel_size=3, strides=1, padding='same', activation='relu')(b1)
    # branch2
    b2 = Conv2D(filters=int(num_filters/4), kernel_size=1, strides=1, padding='same', activation='relu')(x)
    b2 = Conv2D(filters=num_filters, kernel_size=3, strides=1, padding='same', activation='relu')(b2)
    # branch3
    b3 = MaxPool2D(pool_size=2, strides=1, padding='same')(x)
    b3 = Conv2D(filters=num_filters, kernel_size=1, strides=1, padding='same', activation='relu')(b3)
    # branch4
    b4 = Conv2D(filters=num_filters, kernel_size=1, strides=1, padding='same', activation='relu')(x)

    # concat
    out = concatenate([b1, b2, b3, b4], axis=3)
    return out


def inception_tiny():
    input = Input(shape=[32, 32, 3])
    x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(input)
    x = BatchNormalization()(x)
    # stage 0 :  fm_size 32
    x = Inception_Block_v1(x, num_filters=64)
    x = Inception_Block_v1(x, num_filters=64)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    # stage 1 :  fm_size 16
    x = Inception_Block_v1(x, num_filters=128)
    x = Inception_Block_v1(x, num_filters=128)
    x = Inception_Block_v1(x, num_filters=128)
    x = MaxPool2D(pool_size=2, strides=2)(x)
    # stage 2 : fm_size 8
    x = Inception_Block_v1(x, num_filters=256)
    x = Inception_Block_v1(x, num_filters=256)
    x = Inception_Block_v1(x, num_filters=256)
    # output
    x = GlobalAveragePooling2D()(x)
    output = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=[input],
                  outputs=[output])

    return model




if __name__ == "__main__":
    model = inception_tiny()
    model.summary()
    plot_model(model, to_file='inception_tiny.png')
    #model = inception_v1()
