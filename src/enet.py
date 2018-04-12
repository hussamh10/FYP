import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras import backend as keras 
from time import time

from generator import generateENET as generate
from generator import generateENETRandom as generateRandom
from generator import countFolderImages

"""
+++++++++++++++++++++++++++++++++++++++++++++++++++
Notes: 
        - Added Wasserstein loss function
        - Removed Wasserstein
+++++++++++++++++++++++++++++++++++++++++++++++++++
"""

def wasserstein_loss(y_true, y_pred):

    """Calculates the Wasserstein loss for a sample batch.
    The Wasserstein loss function is very simple to calculate. In a standard GAN, the discriminator
    has a sigmoid output, representing the probability that samples are real or generated. In Wasserstein
    GANs, however, the output is linear with no activation function! Instead of being constrained to [0, 1],
    the discriminator wants to make the distance between its output for real and generated samples as large as possible.
    The most natural way to achieve this is to label generated samples -1 and real samples 1, instead of the
    0 and 1 used in normal GANs, so that multiplying the outputs by the labels will give you the loss immediately.
    Note that the nature of this loss means that it can be (and frequently will be) less than 0."""

    return K.mean(y_true * y_pred)

class myUnet(object):


        def __init__(self, img_rows, img_cols):

                self.img_rows = img_rows
                self.img_cols = img_cols

        def get_unet(self):
                inputs = Input((self.img_rows, self.img_cols,2))
                conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
                print ("conv1 shape:",conv1.shape)
                conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
                print ("conv1 shape:",conv1.shape)
                pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
                print ("pool1 shape:",pool1.shape)

                conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
                print ("conv2 shape:",conv2.shape)
                conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
                print ("conv2 shape:",conv2.shape)
                pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
                print ("pool2 shape:",pool2.shape)

                conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
                print ("conv3 shape:",conv3.shape)
                conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
                print ("conv3 shape:",conv3.shape)
                pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
                print ("pool3 shape:",pool3.shape)

                conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
                conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
                drop4 = Dropout(0.5)(conv4)
                pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

                conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
                conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
                drop5 = Dropout(0.5)(conv5)

                up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
                merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
                conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
                conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

                up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
                merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
                conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
                conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

                up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
                merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
                conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
                conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

                up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
                merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
                conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
                conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
                conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
                conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

                model = Model(input = inputs, output = conv10)

                model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy' , metrics = ['accuracy'])
                return model


        def train(self, folders_limit, main, frame_pre, frame_ext, t):

                TensorBoard(log_dir='..\\graphs\\' + 'GraphENET'+t, histogram_freq=0, write_graph=True, write_images=True)
                tbCallBack = TensorBoard(log_dir='..\\graphs\\' + 'GraphENET'+t, histogram_freq=0, write_graph=True, write_images=True)

                model = self.get_unet()

                checkpoint_parent = '..\\checkpoints\\'
                checkpoint = checkpoint_parent + 'enet_' + t + '.hdf5'
                best_checkpoint = checkpoint_parent + 'best_enet' + t + '.hdf5'
                model_checkpoint = ModelCheckpoint(checkpoint, monitor='loss', save_best_only=False, verbose=1, mode='auto', period=100)
                mc_best = ModelCheckpoint(best_checkpoint, monitor='loss', save_best_only=True, verbose=1, mode='auto' , period=100)

                model.fit_generator(generateRandom(14000, folders_limit, main, frame_pre, frame_ext), steps_per_epoch=30, epochs=105, verbose=1, callbacks=[model_checkpoint, tbCallBack, mc_best])

def get_unet():
        myunet = myUnet(224, 224)
        return myunet.get_unet()

if __name__ == '__main__':
        main = '..\\data\\'
        folders = 2
        
        
        countFolderImages(folders,main)
		
        myunet = myUnet(224, 224)
        #(number of folders, directory where the folders are, pre(ignore this), extension, time)
        myunet.train(folders, main, '', '.jpg', str(time()))
