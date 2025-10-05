import tensorflow as tf
from tensorflow.keras import layers, models

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
class DCGAN:
    def __init__(self, latent_dim=128, img_shape=(32, 32, 3), d_lr=2e-4, g_lr=2e-4, beta_1=0.5, beta_2=0.999):
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        # Modelos
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()

        # Compilar discriminador (se entrena solo al entrenar el D)
        self.discriminator.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=d_lr, beta_1=beta_1, beta_2=beta_2),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        # Modelo combinado (G engaña a D): congelamos D
        self.discriminator.trainable = False
        z = layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)
        validity = self.discriminator(img)
        self.combined = models.Model(z, validity, name="gan_combined")
        self.combined.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=g_lr, beta_1=beta_1, beta_2=beta_2),
            loss='binary_crossentropy'
        )

    def _build_generator(self):
        """Generador DCGAN para 32x32x3."""
        model = models.Sequential(name='generator')
        n = 256  # canales base

        model.add(layers.Input(shape=(self.latent_dim,)))
        model.add(layers.Dense(4*4*n*2, use_bias=False))
        model.add(layers.Reshape((4, 4, n*2)))  # (4,4,512)

        # Bloques upsampling: 4->8->16->32
        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Conv2DTranspose(n, kernel_size=4, strides=2, padding='same', use_bias=False))  # 8x8

        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Conv2DTranspose(n//2, kernel_size=4, strides=2, padding='same', use_bias=False))  # 16x16

        model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        model.add(layers.Conv2DTranspose(n//4, kernel_size=4, strides=2, padding='same', use_bias=False))  # 32x32

        # Capa de salida a 3 canales con tanh
        model.add(layers.Conv2D(self.img_shape[2], kernel_size=3, padding='same', activation='tanh'))

        return model

    def _build_discriminator(self):
        """Discriminador DCGAN para 32x32x3."""
        model = models.Sequential(name='discriminator')
        n = 64  # canales base

        model.add(layers.Input(shape=self.img_shape))
        # 32->16
        model.add(layers.Conv2D(n, kernel_size=4, strides=2, padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))

        # 16->8
        model.add(layers.Conv2D(n*2, kernel_size=4, strides=2, padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))

        # 8->4
        model.add(layers.Conv2D(n*4, kernel_size=4, strides=2, padding='same'))
        model.add(layers.LeakyReLU(alpha=0.2))
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))  # prob. de "real"

        return model

