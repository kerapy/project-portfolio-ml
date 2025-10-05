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

################ para utilizar el modelo guardado ################
from keras.models import load_model
from numpy.random import randn
from matplotlib import pyplot
 
# generamos puntos en el espacio latente
def generate_latent_points(latent_dim, n_samples):
	x_input = randn(latent_dim * n_samples)
	# ajustamos la forma al modelo
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
# plot the generated images
def create_plot(examples, n):
	# graficamos las imagenes con pyplot
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(examples[i, :, :])
	pyplot.show()
 

######################## utils para lstm series temporales ###########################
import numpy as np
import matplotlib.pyplot as plt
from data import gen_sequence

def plot_history(history):
    plt.figure(figsize=(8,4))
    for k in ("loss","val_loss","accuracy","val_accuracy"):
        if k in history.history:
            plt.plot(history.history[k], label=k)
    plt.legend(); plt.xlabel("Epoch"); plt.title("Training curves")
    plt.tight_layout(); plt.show()

def prob_failure(model, df_test, machine_id, seq_length, feature_cols, id_col="id"):
    """
    Devuelve la probabilidad (%) de fallo de la ÚLTIMA ventana.
    """
    machine_df = df_test[df_test[id_col] == machine_id]
    seqs = gen_sequence(machine_df, seq_length, feature_cols)
    if seqs.size == 0:
        raise ValueError("Esa máquina no tiene suficientes filas para la ventana indicada.")
    preds = model.predict(seqs, verbose=0).reshape(-1)
    return float(preds[-1] * 100.0)
