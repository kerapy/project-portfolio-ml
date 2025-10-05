
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
def load_cifar10_for_gan():
    """
    Carga CIFAR-10 y preprocesa para DCGAN
    """
    (X_train, _), (_, _) = cifar10.load_data()
    
    # Normalizamos a [-1, 1] (importante para GANs con tanh)
    X_train = X_train.astype('float32')
    X_train = (X_train - 127.5) / 127.5
    
    print(f"CIFAR-10 cargado: {X_train.shape}")
    print(f"Rango de valores: [{X_train.min():.2f}, {X_train.max():.2f}]")
    
    return X_train

def normalize_images(images):
    """
    Normaliza imágenes de [0, 255] a [-1, 1]
    """
    images = images.astype('float32')
    images = (images - 127.5) / 127.5
    return images


def denormalize_images(images):
    """
    Desnormaliza imágenes de [-1, 1] a [0, 1]
    (útil para visualización)
    """
    images = 0.5 * images + 0.5
    images = np.clip(images, 0, 1)
    return images