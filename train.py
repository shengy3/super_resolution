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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from keras.models import load_model
from scipy.ndimage import zoom

np.random.seed(10)
image_shape = (8,8,1)
downscale_factor = 2


df = pd.read_pickle("./data/2.5TeV_neutrons_UniPlane22_Reposite_100k_55file.pickle")
df.head()

#load data
lr = df['4_truth']
hr = df[f'{image_shape[0]}_truth']


def scale_img(img, axis):
    return (img - np.mean(img))/np.std(img)

lr = np.stack(lr.apply(scale_img, axis = 1).values)
hr = np.stack(hr.apply(scale_img, axis = 1).values)

x_train_lr, x_test_lr, x_train_hr, x_test_hr = train_test_split(lr,hr, test_size=0.33, random_state=42)



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

def plot_generated_images(epoch,generator, examples=3 , dim=(1, 3), figsize=(20, 5)):
    
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
    
    plt.subplot(dim[0], dim[1], 1)
    plt.imshow(image_batch_lr[1].reshape(4,4), interpolation='nearest')
    plt.axis('off')
        
    plt.subplot(dim[0], dim[1], 2)
    plt.imshow(generated_image[1].reshape(image_shape[0],image_shape[1]), interpolation='nearest')
    plt.axis('off')
    
    plt.subplot(dim[0], dim[1], 3)
    plt.imshow(zoom(image_batch_lr[1].reshape(4,4), downscale_factor).reshape(image_shape[0],image_shape[1]), interpolation='nearest')
    plt.axis('off')

    plt.subplot(dim[0], dim[1], 4)
    plt.imshow(image_batch_hr[1], interpolation='nearest')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('./out/gan_generated_image_epoch_%d.png' % epoch)
    

def train(epochs=1, batch_size=128):

    
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
            if e == 0 and os.isfile("./checkpoint/gen_model100.h5"):
                print("Continue training")
                generator = load_model('./checkpoint/gen_model100.h5' % e)
                discriminator = load_model('./checkpoint/dis_model100.h5' % e)
                gan = load_model('./checkpoint/gan_model100.h5' % e)
            
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
            generator.save('./checkpoint/gen_model%d.h5' % e)
            discriminator.save('./checkpoint/dis_model%d.h5' % e)
            gan.save('./checkpoint/gan_model%d.h5' % e)
    print("Done")

train(epochs=100, batch_size=128)
