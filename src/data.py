import pandas as pd
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


########################### lstm series temporales ###########################
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Columnas (mismo orden Kaggle)
# -----------------------------
COLS = ['id','cycle','setting1','setting2','setting3'] + [f's{i}' for i in range(1,22)]
N_EXPECTED = len(COLS)  # 26

def _read_pm_txt(path: str) -> pd.DataFrame:
    # Algunos dumps traen columnas vacías; nos quedamos con las primeras 26
    df = pd.read_csv(path, sep=r"\s+", header=None)
    if df.shape[1] > N_EXPECTED:
        df = df.iloc[:, :N_EXPECTED]
    df.columns = COLS
    return df

# --------------------------------------
# Carga 3 archivos y construye RUL/label
# --------------------------------------
def load_pm_triplet_to_classification(
    train_path: str,
    test_path: str,
    truth_path: str,
    horizon: int = 30,
):
    """
    Devuelve: df_train, df_test (con label_bc), features_col_name
    """
    dataset_train = _read_pm_txt(train_path)
    dataset_test  = _read_pm_txt(test_path)

    pm_truth = pd.read_csv(truth_path, sep=r"\s+", header=None)
    # La 2a col puede ser vacía; nos quedamos con la primera
    pm_truth = pm_truth.iloc[:, :1]
    pm_truth.columns = ["more"]
    pm_truth["id"] = pm_truth.index + 1

    # ---- test: ttf a partir de truth + max cycle por id 
    rul = dataset_test.groupby("id")["cycle"].max().reset_index().rename(columns={"cycle":"max"})
    pm_truth["rtf"] = pm_truth["more"] + rul["max"]
    dataset_test = dataset_test.merge(pm_truth[["id","rtf"]], on="id", how="left")
    dataset_test["ttf"] = dataset_test["rtf"] - dataset_test["cycle"]
    dataset_test = dataset_test.drop(columns=["rtf"])

    # ---- train: ttf = max(cycle,id) - cycle
    dataset_train["ttf"] = dataset_train.groupby("id")["cycle"].transform("max") - dataset_train["cycle"]

    # ---- labels binarias
    df_train = dataset_train.copy()
    df_test  = dataset_test.copy()
    df_train["label_bc"] = (df_train["ttf"] <= horizon).astype("int32")
    df_test["label_bc"]  = (df_test["ttf"]  <= horizon).astype("int32")

    features_col_name = ['setting1','setting2','setting3'] + [f's{i}' for i in range(1,22)]
    return df_train, df_test, features_col_name

# --------------------------------------
# Escalado MinMax
# --------------------------------------
def minmax_fit_transform(df_train: pd.DataFrame, df_test: pd.DataFrame, feature_cols):
    scaler = MinMaxScaler()
    df_train = df_train.copy()
    df_test  = df_test.copy()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols]  = scaler.transform(df_test[feature_cols])
    return df_train, df_test, scaler

# --------------------------------------
# Generadores con PADDING de ceros
# --------------------------------------
def gen_sequence(id_df: pd.DataFrame, seq_length: int, seq_cols):
    df_zeros = pd.DataFrame(np.zeros((seq_length - 1, id_df.shape[1])), columns=id_df.columns)
    id_df_pad = pd.concat([df_zeros, id_df], ignore_index=True)
    data_array = id_df_pad[seq_cols].values
    n = data_array.shape[0]
    out = []
    for start, stop in zip(range(0, n - seq_length), range(seq_length, n)):
        out.append(data_array[start:stop, :])
    return np.asarray(out, dtype=np.float32)

def gen_label(id_df: pd.DataFrame, seq_length: int, seq_cols, label_col: str):
    df_zeros = pd.DataFrame(np.zeros((seq_length - 1, id_df.shape[1])), columns=id_df.columns)
    id_df_pad = pd.concat([df_zeros, id_df], ignore_index=True)
    n = id_df_pad.shape[0]
    y = []
    for stop in range(seq_length, n):
        y.append(id_df_pad[label_col].iloc[stop])
    return np.asarray(y, dtype=np.int32)

def make_X_y(df: pd.DataFrame, id_col: str, feature_cols, label_col: str, seq_length: int):
    X_list, y_list = [], []
    for id_val, g in df.groupby(id_col):
        X_id = gen_sequence(g, seq_length, feature_cols)
        if X_id.size == 0:
            continue
        y_id = gen_label(g, seq_length, feature_cols, label_col)
        m = min(len(X_id), len(y_id))
        X_list.append(X_id[:m])
        y_list.append(y_id[:m])
    if not X_list:
        n_feat = len(feature_cols)
        return np.empty((0, seq_length, n_feat), dtype=np.float32), np.empty((0,), dtype=np.int32)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

####### adquisicion datos transformer imdb #########
from keras.datasets import imdb
def get_imdb_data(num_words=10000):
    (X_train, y_train),(X_test,y_test)= imdb.load_data(num_words=num_words)
    return X_train, y_train, X_test, y_test