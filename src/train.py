#Entrenamiento se realizara en el ipynb
####### GAN##########
import numpy as np
from data import  generate_real_samples, generate_fake_samples, generate_latent_points
from utils import save_plot
import os

def summarize_performance(epoch, g_model, d_model, dataset, latent_dim,
                          n_samples=150, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    # reales
    X_real, y_real = generate_real_samples(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    # falsas
    X_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(X_fake, y_fake, verbose=0)
    print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
    # plot (formato tutorial)
    save_plot(X_fake, epoch, n=7, save_dir=out_dir)
    # guardar generador 
    filename = os.path.join(out_dir, f'generator_model_{epoch+1:03d}.keras')
    g_model.save(filename)
    print(f'Guardado {filename}')

def train_gan(gan, dataset, n_epochs=200, n_batch=128, out_dir="results"):
    """
    - Recorre TODOS los batches del dataset en cada epoch.
    - Cada 10 epochs: summarize_performance + save_plot + save generator.
    """
    g_model, d_model, gan_model = gan.generator, gan.discriminator, gan.combined
    latent_dim = gan.latent_dim

    bat_per_epo = int(dataset.shape[0] / n_batch)
    half_batch = int(n_batch / 2)

    for i in range(n_epochs):
        for j in range(bat_per_epo):
            # 1) reales
            X_real, y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = d_model.train_on_batch(X_real, y_real)
            # 2) falsas
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
            # 3) generador (etiquetas invertidas)
            X_gan = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            print(f'>{i+1}, {j+1}/{bat_per_epo}, d1={d_loss1:.3f}, d2={d_loss2:.3f} g={g_loss:.3f}', end='\r')

        print()  # salto de línea al cerrar la época
        # cada 10 epochs: evaluar y guardar
        if (i + 1) % 10 == 0:
            summarize_performance(i, g_model, d_model, dataset, latent_dim, out_dir=out_dir)



######################## lstm ############################################

import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def get_callbacks(monitor="val_loss", patience=3):
    #EarlyStopping(patience=3)
    return [tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True)]

def train(model, X_train, y_train, epochs=10, batch_size=200, validation_split=0.05,
          callbacks=None, class_weight=None, verbose=1):
    cbs = callbacks or get_callbacks("val_loss", patience=0)
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=verbose,
        callbacks=cbs,
        class_weight=class_weight
    )
    return history

def evaluate_on_arrays(model, X, y, batch_size=200):
    y_prob = model.predict(X, batch_size=batch_size, verbose=0).reshape(-1)
    y_pred = (y_prob >= 0.5).astype("int32")
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred).tolist()
    report = classification_report(y, y_pred, output_dict=True)
    return {"accuracy": float(acc), "confusion_matrix": cm, "report": report}

def predict_classes(model, X, batch_size=200):
    # Reemplazo de predict_classes()
    y_prob = model.predict(X, batch_size=batch_size, verbose=0).reshape(-1)
    return (y_prob >= 0.5).astype("int32")
