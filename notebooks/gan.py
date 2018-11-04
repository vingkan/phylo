from __future__ import print_function, division

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

import sys

import numpy as np
import phylo as ph

import tensorflow
print(tensorflow.__version__)

class GAN(object):

    def __init__(self):
        self.shape = ph.M ** 2
        self.discriminator = None # Discriminator
        self.generator = None # Shareholder value generator
        self.adversary_model = None # Adversarial Model
        self.discriminator_model = None # Discriminator Model

    def discriminate(self):
        '''
        Discriminator that transforms 96*96 flat list to a source of truth
        '''
        if self.discriminator:
            return self.discriminator

        self.discriminator = Sequential()
        depth = 64
        dropout = 0.4
        channel = 1

        dim = int(self.shape ** .5)

        # In: 96*96 flat list
        # Out: 96 x 96 x 1
        self.discriminator.add(Dense(dim*dim*channel, input_dim=self.shape))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Reshape((dim, dim, channel)))
        self.discriminator.add(Dropout(dropout))

        # In: dim*dim*channel
        # Out: 48 x 48 x 1, depth = 64
        self.discriminator.add(Conv2D(depth*1, 5, strides=2, padding='same'))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dropout(dropout))

        self.discriminator.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dropout(dropout))

        self.discriminator.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dropout(dropout))

        self.discriminator.add(Conv2D(depth*8, 5, strides=2, padding='same'))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dropout(dropout))

        self.discriminator.add(Conv2D(depth*16, 5, strides=2, padding='same'))
        self.discriminator.add(LeakyReLU(alpha=0.2))
        self.discriminator.add(Dropout(dropout))

        # Flatten back out
        self.discriminator.add(Flatten())
        self.discriminator.add(Dense(1))
        self.discriminator.add(Activation('sigmoid'))
        self.discriminator.summary()
        return self.discriminator

    def generate(self):
        '''
        Generator that takes random noise and produces an image from it
        '''
        if self.generator:
            return self.generator

        self.generator = Sequential()
        dropout = 0.4
        depth = 64 * 16
        dim = 3

        self.generator.add(Dense(dim*dim*depth, input_dim=self.shape))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))
        self.generator.add(Reshape((dim, dim, depth)))
        self.generator.add(Dropout(dropout))

        self.generator.add(UpSampling2D())
        self.generator.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))

        self.generator.add(UpSampling2D())
        self.generator.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))

        self.generator.add(UpSampling2D())
        self.generator.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))

        self.generator.add(UpSampling2D())
        self.generator.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))

        self.generator.add(UpSampling2D())
        self.generator.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.generator.add(BatchNormalization(momentum=0.9))
        self.generator.add(Activation('relu'))

        # Get back to 96*96 Flat list
        self.generator.add(Conv2DTranspose(1, 5, padding='same'))
        self.generator.add(Flatten())
        self.discriminator.add(Activation('sigmoid'))
        self.generator.summary()
        return self.generator

    def build_dm(self):
        if self.discriminator_model:
            return self.discriminator_model

        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.discriminator_model = Sequential()
        self.discriminator_model.add(self.discriminate())
        self.discriminator_model.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])

        return self.discriminator_model

    def build_am(self):
        if self.adversary_model:
            return self.adversary_model

        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.adversary_model = Sequential()
        self.adversary_model.add(self.generate())
        self.adversary_model.add(self.discriminate())
        self.adversary_model.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])

        return self.adversary_model

class Manager(object):

    def __init__(self):
        self.pokemon = ph.vectorize_pokemon(ph.REGULAR_POKEMON_PATH) + ph.vectorize_pokemon(ph.SHINY_POKEMON_PATH)
        self.GAN = GAN()
        self.discriminator = self.GAN.build_dm()
        self.adversarial_model = self.GAN.build_am()
        self.generator = self.GAN.generate()

    def train(self, train_steps=2000, batch_size=256, save_interval=0):

        for i in range(train_steps):
            print("Training...#{}".format(i))
            images_train = np.random.choice(range(len(self.pokemon)), min(batch_size, len(self.pokemon)))
            images_train = np.asarray([self.pokemon[i] for i in images_train])
            noise = np.asarray([ph.generate_random() for i in range(batch_size)])
            print("Generating fake images")
            fake_images = self.generator.predict(noise)
            X = np.concatenate((images_train, fake_images))
            Y = np.ones([2*batch_size, 1])
            Y[batch_size:, :] = 0
            print("The right is winning")
            d_loss = self.discriminator.train_on_batch(X, Y)

            Y = np.ones([batch_size, 1])
            noise = np.asarray([ph.generate_random() for i in range(batch_size)])
            a_loss = self.adversarial_model.train_on_batch(noise, Y)

            log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)

            if save_interval>0:
                if (i+1)%save_interval==0:
                    self.save_image(i)

    def save_image(self, save2File=False, samples=16, fake=True, iter_):
        if fake:
            noise = np.asarray([ph.generate_random() for i in range(samples)])
            images = self.generator.predict(noise)
            images = np.asarray([[abs(xj * 4) % 5 for xj in xi] for xi in images]).astype(int)
            cnt = 0
            for img in images:
                ph.save_image(img, "iteration_{}/fake_img_{}.png".format(iter_, cnt))
                cnt += 1

if __name__ == "__main__":
    manager = Manager()
    manager.train(train_steps=30000, batch_size=256, save_interval=500)
