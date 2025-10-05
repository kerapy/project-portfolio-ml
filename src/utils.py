#------------------ utilidades para cnn cifar10 ----------------#

import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import plot_model
import numpy as np
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


#------------------ utilidades para gan cifar10 ----------------#
def save_plot(examples, epoch, n=7, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    examples = (examples + 1) / 2.0  # [-1,1] -> [0,1]
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)
        plt.axis('off')
        plt.imshow(examples[i])
    filename = os.path.join(save_dir, f'generated_plot_e{epoch+1:03d}.png')
    plt.savefig(filename)
    plt.close()
    return filename

def show_grid(examples, n=5):
    """Muestra un grid de imágenes generadas sin guardar."""
    if examples.min() < 0:
        examples = (examples + 1) / 2.0
    examples = np.clip(examples, 0.0, 1.0)

    fig, axs = plt.subplots(n, n, figsize=(n, n))
    idx = 0
    for i in range(n):
        for j in range(n):
            axs[i, j].imshow(examples[idx])
            axs[i, j].axis("off")
            idx += 1
    plt.show()

def save_model_plots(generator, discriminator, out_dir="results", dpi=110):
    """Guarda diagrams de Generator y Discriminator usando Graphviz/pydot."""
    os.makedirs(out_dir, exist_ok=True)
    gen_path = os.path.join(out_dir, "generator.png")
    disc_path = os.path.join(out_dir, "discriminator.png")
    plot_model(generator, to_file=gen_path, show_shapes=True, show_layer_names=True, dpi=dpi)
    plot_model(discriminator, to_file=disc_path, show_shapes=True, show_layer_names=True, dpi=dpi)

    return gen_path, disc_path