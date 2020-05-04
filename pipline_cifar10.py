import tensorflow as tf
from tensorflow import keras
import PIL.Image as Image
import numpy as np
import os
import time

from nets.VGG import vgg16
from nets.ResNet import resnet_v1, resnet_v2

from tensorflow.keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_cifar10_data(path='cifar-10-batches-py'):
    from tensorflow.python.keras.datasets.cifar import load_batch
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))


    x_train = x_train.transpose(0, 2, 3, 1)
    x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)


def lr_schedule(epoch):
    return 0.01

if __name__ == "__main__":
    # params
    nets = ['vgg16', 'res18_v1', 'res18_v2', 'inception_v1']
    net = nets[0]

    num_classes = 10
    batchsize = 32
    lr = 0.01
    lr_decay = 1e-6
    lr_drop = 20
    epochs = 30
    initial_epoch = 0
    train_csv = './callbacks/csvs/cifar10_{}.csv'.format(net)
    log_dir = './callbacks/logs/cifar10_{}'.format(net)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    ckpt_dir = './callbacks/ckpts/{}/'.format(net)
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    subtract_pixel_mean = True
    data_aug = True

    # 1. parse cifar 10 and build data pipline
    (x_train, y_train), (x_test, y_test) = load_cifar10_data(path='../data/data68')
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
    # one-hot
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    #cifar10_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batchsize)
    cifar10_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batchsize)
    # data augmentation
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # 2. load models

    if net == 'vgg16':
        model = vgg16()
    elif net == 'res18_v1':
        model = resnet_v1([32, 32, 3], 20)
    elif net == 'res18_v2':
        model = resnet_v2([32, 32, 3], 20)

    model.summary()

 #   adam = Adam(learning_rate=lr, decay=lr_decay, momentum=0.9)
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


    # 3. add call backs
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-5, patience=3, restore_best_weights=True)
    cvs_logger = CSVLogger(filename=train_csv, separator=',', append=False)
    model_ckpt = ModelCheckpoint(filepath=ckpt_dir+'cifar10.{epoch:02d}-{val_loss:.2f}.hdf5', save_best_only=True, save_freq='epoch')
    lr_scheduler = LearningRateScheduler(schedule=lr_schedule)
    tensorboard = TensorBoard(log_dir=log_dir)

    callbacks = [early_stopping,
                 cvs_logger,
                 model_ckpt,
                 lr_scheduler,
                 tensorboard]


    # 4. train
    start = time.clock()

    history = model.fit(x=datagen.flow(x_train, y_train, batch_size=batchsize),
                        epochs=epochs,
                        callbacks=callbacks,
                        validation_data=cifar10_test,
                        shuffle=True,
                        initial_epoch=initial_epoch,
                        validation_freq=1)

    elapsed = (time.clock() - start)
    print("Time used:", elapsed)













