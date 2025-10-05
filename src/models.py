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
        # compile model
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
        # output con activacion tanh
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

    # opcional
    def generate_noise(self, n_samples: int):
        return np.random.randn(n_samples, self.latent_dim)
    

########## lstm modelo ########################################33
import tensorflow as tf
from tensorflow.keras import layers, models

class LSTMSimpleClassifier:
    """
      LSTM(100, return_sequences=True) -> Dropout(0.2)
      LSTM(50) -> Dropout(0.2)
      Dense(1, sigmoid)
    """
    def __init__(self, timestamp: int, nb_features: int, lr=1e-3):
        self.model = self._build(timestamp, nb_features, lr)

    def _build(self, timestamp, nb_features, lr):
        m = models.Sequential(name="LSTM_Classifier")
        m.add(layers.Input(shape=(timestamp, nb_features)))
        m.add(layers.LSTM(100, return_sequences=True))
        m.add(layers.Dropout(0.2))
        m.add(layers.LSTM(50, return_sequences=False))
        m.add(layers.Dropout(0.2))
        m.add(layers.Dense(1, activation="sigmoid"))
        m.compile(
            loss="binary_crossentropy",
            optimizer=tf.keras.optimizers.Adam(lr),
            metrics=["accuracy"]
        )
        return m

    def summary(self):
        return self.model.summary()


##################### transformer modelo ##############################

#### embedding ####
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import ops

class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = ops.shape(inputs)[-1]
        positions = ops.arange(0,length,1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return ops.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim,
        })
        return config
    
##### Encoder ######
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dense_dim = dense_dim
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config

#### armando el modelo

def build_transformer_model(
        vocab_size,
        sequence_length,
        embed_dim,
        num_heads,
        dense_dim):
    inputs = keras.Input(shape=(None,), dtype="int64")
    x = PositionalEmbedding(sequence_length,vocab_size, embed_dim)(inputs)
    x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", # Cambiar a optimizador Adam
                loss="binary_crossentropy",
                metrics=["accuracy"])
    model.summary()
    return model