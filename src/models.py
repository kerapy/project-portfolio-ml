import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (Dense, Reshape, Flatten,
                                     Conv2D, Conv2DTranspose,
                                     LeakyReLU, Dropout)
from tensorflow.keras.optimizers import Adam
import numpy as np
class CnnModel:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build()

    def build(self):
        #Construimos el modelo aqui
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            #capa1 convolucion + relu + maxpooling
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            #capa2 convolucion + relu + maxpooling
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            #capa3 convolucion + relu
            layers.Conv2D(128, (3, 3), activation='relu'),
            #capa de aplanamiento
            layers.Flatten(),
            #capa fully connected con 64 neuronas + relu
            layers.Dense(64, activation='relu'),
            #capa de salida con softmax
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
        
    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print("Modelo compilado exitosamente.")

    def summary(self):
        print("---------------Resumen del modelo:--------------------")
        self.model.summary()
        print("------------------------------------------------------")

# Creamos nuestro modelo GAN para cifar10
class GAN:
    """
    DCGAN para CIFAR-10 basado en el tutorial de Machine Learning Mastery.
    Encapsula: generator, discriminator y el modelo combinado para entrenar G.
    """
    def __init__(self, latent_dim: int = 100, img_shape=(32, 32, 3)) -> None:
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.discriminator = self._build_discriminator()
        self.generator = self._build_generator()
        self.combined = self._build_gan(self.generator, self.discriminator)

    # --------- Discriminator  ---------
    def _build_discriminator(self) -> Sequential:
        model = Sequential(name="Discriminator")
        # normal
        model.add(Conv2D(64, (3,3), padding='same', input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # downsample
        model.add(Conv2D(256, (3,3), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # classifier
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        # compile (igual que el tutorial)
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    # --------- Generator  ---------
    def _build_generator(self) -> Sequential:
        model = Sequential(name="Generator")
        # foundation for 4x4 image
        n_nodes = 256 * 4 * 4
        model.add(Dense(n_nodes, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((4, 4, 256)))
        # upsample 8x8
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample 16x16
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # upsample 32x32
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        # output
        model.add(Conv2D(3, (3,3), activation='tanh', padding='same'))
        return model

    # --------- GAN combinado ---------
    def _build_gan(self, g_model: Sequential, d_model: Sequential) -> Sequential:
        d_model.trainable = False
        model = Sequential(name="Combined_GAN")
        model.add(g_model)
        model.add(d_model)
        opt = Adam(learning_rate=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    # helper opcional
    def generate_noise(self, n_samples: int):
        return np.random.randn(n_samples, self.latent_dim)