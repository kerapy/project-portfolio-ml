
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
def load_cifar10():
    '''Carga el dataset CIFAR-10.'''
    (train_images, train_labels),(test_images,test_labels) = cifar10.load_data()
    return train_images, train_labels, test_images, test_labels

def preprocess_data(images, labels, num_classes):
    '''Preprocesa las imagenes y etiquetas.'''
    images = images.astype('float32') / 255.0
    labels = to_categorical(labels, num_classes)
    return images, labels

def get_data_shape(data):
    '''Retorna las formas de los datos.'''
    return data.shape

def get_class_names():
    '''Retorna los nombres de las clases en CIFAR-10.'''
    return ['avion','automovil','pajaro','gato','ciervo','perro','rana','caballo','barco','camion']

def data_augmentation(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True):
    '''Retorna un generador de aumentacion de datos.'''
    datagen = ImageDataGenerator(
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        horizontal_flip=horizontal_flip,
    )
    return datagen


#---------------- GAN ----------------#
def load_real_samples():
    (trainX, _), (_, _) = cifar10.load_data()
    X = trainX.astype('float32')
    X = (X - 127.5) / 127.5  # [-1,1]
    print(X.shape)
    return X

def generate_real_samples(dataset, n_samples):
    ix = np.random.randint(0, dataset.shape[0], n_samples)
    X = dataset[ix]
    y = np.ones((n_samples, 1))
    return X, y

def generate_latent_points(latent_dim, n_samples):
    x_input = np.random.randn(latent_dim * n_samples)
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

def generate_fake_samples(g_model, latent_dim, n_samples):
    x_input = generate_latent_points(latent_dim, n_samples)
    X = g_model.predict(x_input, verbose=0)
    y = np.zeros((n_samples, 1))
    return X, y