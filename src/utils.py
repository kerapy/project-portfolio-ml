#------------------ utilidades para cnn cifar10 ----------------#

import matplotlib.pyplot as plt

def plot_training_curves(history):
    """
    Grafica la precisión y la pérdida de entrenamiento y validación
    a partir del history de Keras.
    """
    plt.figure(figsize=(12,6))
    # Precisión
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Precisión de validación')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()
    plt.title('Curva de Precisión')
    # Pérdida
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.title('Curva de Pérdida')
    plt.tight_layout()
    plt.show()