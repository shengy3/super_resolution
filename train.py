from network import Generator, Discriminator

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras.applications.vgg19 import VGG19
from keras.layers.convolutional import UpSampling2D
from keras.models import Model
from keras.optimizers import SGD, Adam, RMSprop
import keras
import keras.backend as K
from keras.layers import Lambda, Input
import tensorflow as tf
#import skimage.transform
#from skimage import data, io, filters
import numpy as np
from numpy import array
#from skimage.transform import rescale, resize
#from scipy.misc import imresize
import os

np.random.seed(10)
image_shape = (8,8,1)

def vgg_loss(y_true, y_pred):
    
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    loss_model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    loss_model.trainable = False
    return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def get_gan_network(discriminator, shape, generator, optimizer):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(inputs=gan_input, outputs=[x,gan_output])
    gan.compile(loss=["mae", "binary_crossentropy"],
                loss_weights=[1., 1e-3],
                optimizer=optimizer)

    return gan


print("data processed")

def plot_generated_images(epoch,generator, examples=3 , dim=(1, 3), figsize=(15, 5)):
    
    rand_nums = np.random.randint(0, x_test_hr.shape[0], size=examples)
    image_batch_hr = x_test_hr[rand_nums]
    image_batch_lr = x_test_lr[rand_nums].reshape(-1,4,4,1)
    gen_img = generator.predict(image_batch_lr)
    """
    #generated_image = denormalize(gen_img)
    #image_batch_lr = denormalize(image_batch_lr)
    
    """
    generated_image = gen_img
    image_batch_lr = image_batch_lr
    #generated_image = deprocess_HR(generator.predict(image_batch_lr))
    
    plt.figure(figsize=figsize)
    
    img_1 = plt.subplot(dim[0], dim[1], 1)
    plt.colorbar(img_1)
    plt.imshow(image_batch_lr[1].reshape(4,4), interpolation='nearest')
    plt.axis('off')
        
    img_2 = plt.subplot(dim[0], dim[1], 2)
    plt.colorbar(img_2)
    plt.imshow(generated_image[1].reshape(8,8), interpolation='nearest')
    plt.axis('off')
    
    img3 = plt.subplot(dim[0], dim[1], 3)
    plt.colorbar(img3)
    plt.imshow(image_batch_hr[1], interpolation='nearest')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('../out/gan_generated_image_epoch_%d.png' % epoch)
    

def train(epochs=1, batch_size=128):

    downscale_factor = 2
    
    batch_count = int(x_train_hr.shape[0] / batch_size)
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor, image_shape[2])
    generator = Generator(shape).generator()
    discriminator = Discriminator(image_shape).discriminator()

    adam = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    generator.compile(loss='mae', optimizer=adam)
    discriminator.compile(loss="binary_crossentropy", optimizer=adam)
    shape = (image_shape[0]//downscale_factor, image_shape[1]//downscale_factor,  image_shape[2])
    gan = get_gan_network(discriminator, shape, generator, adam)
    #print(gan.summary())

    for e in range(1, epochs+1):
        print ('-'*15, 'Epoch %d' % e, '-'*15)
        for _ in range(batch_count):
            
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            
            image_batch_hr = x_train_hr[rand_nums].reshape(batch_size,image_shape[0],image_shape[1],1)
            image_batch_lr = x_train_lr[rand_nums].reshape(batch_size,4,4,1)
            generated_images_sr = generator.predict(image_batch_lr)

            real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            fake_data_Y = np.random.random_sample(batch_size)*0.2
            
            discriminator.trainable = True
            
            d_loss_real = discriminator.train_on_batch(image_batch_hr, real_data_Y)
            d_loss_fake = discriminator.train_on_batch(generated_images_sr, fake_data_Y)
            #d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            image_batch_hr = x_train_hr[rand_nums].reshape(batch_size,image_shape[0],image_shape[1],1)
            image_batch_lr = x_train_lr[rand_nums].reshape(batch_size,4,4,1)

            gan_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            discriminator.trainable = False
            loss_gan = gan.train_on_batch(image_batch_lr, [image_batch_hr,gan_Y])
            
        print("Loss HR , Loss LR, Loss GAN")
        print(d_loss_real, d_loss_fake, loss_gan)

        if e == 1 or e % 5 == 0:
            plot_generated_images(e, generator)
        if e % 10 == 0:
            generator.save('../checkpoint/gen_model%d.h5' % e)
            discriminator.save('../checkpoint/dis_model%d.h5' % e)
            gan.save('../checkpoint/gan_model%d.h5' % e)
    print("Done")