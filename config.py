# config.py - Refatorado para usar tf.keras.utils.image_dataset_from_directory (TF 2.20+)
import os
import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras import layers
import pathlib

# ===================================================================
# 1. CONSTANTES E CAMINHOS
# ===================================================================

# Variável que deve ser ajustada pelo usuário
DATASET_BASE_PATH = "Jute_Pest_Dataset"

# Dimensões e Parâmetros
STANDARD_IMAGE_SIZE = (224, 224) 
INCEPTION_IMAGE_SIZE = (299, 299)
BATCH_SIZE = 16
NUM_CLASSES = 17 
SEED = 42

# Caminhos para as subpastas
TRAIN_DIR = os.path.join(DATASET_BASE_PATH, 'train')
VALIDATION_DIR = os.path.join(DATASET_BASE_PATH, 'val')
TEST_DIR = os.path.join(DATASET_BASE_PATH, 'test')
MODELS_DIR = 'Modelos'

# Camada de pré-processamento (Normaliza de [0, 255] para [0, 1])
#RESCALE_LAYER = layers.Rescaling(1./255) 

# ===================================================================
# 2. CARREGAMENTO DE DADOS (image_dataset_from_directory)
# ===================================================================

def create_datasets(image_size, batch_size):
    """Cria e retorna os datasets de treino, validação e teste usando o novo método TF."""

    try:
        data_dir_train = pathlib.Path(TRAIN_DIR)
        data_dir_val = pathlib.Path(VALIDATION_DIR)
        data_dir_test = pathlib.Path(TEST_DIR)

        # -------------------------------------------------------------------
        # A. Treino
        # -------------------------------------------------------------------
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir_train,
            labels='inferred',
            label_mode='categorical',
            image_size=image_size,
            batch_size=batch_size,
            shuffle=True,
            seed=SEED,
            interpolation='nearest'
        )
        class_names = train_ds.class_names
        # -------------------------------------------------------------------
        # B. Validação
        # -------------------------------------------------------------------
        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir_val,
            labels='inferred',
            label_mode='categorical',
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False
        )
        
        # -------------------------------------------------------------------
        # C. Teste
        # -------------------------------------------------------------------
        test_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir_test,
            labels='inferred',
            label_mode='categorical',
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False
        )
        
        # Otimização: Cache e Prefetch (Melhora a velocidade de I/O)
        AUTOTUNE = tf.data.AUTOTUNE
        
        train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        print(f"\nDatasets carregados com sucesso. Tamanho: {image_size[0]}x{image_size[1]}")
        return train_ds, val_ds, test_ds, class_names

    except Exception as e:
        print(f"ERRO CRÍTICO: Não foi possível carregar os datasets. Verifique DATASET_BASE_PATH. Erro: {e}")
        return None, None, None


# Inicializa os datasets com o tamanho padrão (224x224) para o cálculo inicial dos pesos
#train_dataset_init, _, _ = create_datasets(STANDARD_IMAGE_SIZE, BATCH_SIZE)
train_dataset_init, validation_dataset_init, test_dataset_init, CLASS_NAMES = create_datasets(STANDARD_IMAGE_SIZE, BATCH_SIZE)

# ===================================================================
# 3. CÁLCULO DE PESOS DE CLASSE (Novo Método)
# ===================================================================

def calculate_class_weights(dataset, num_classes=NUM_CLASSES):
    """Calcula pesos de classe diretamente das imagens no dataset."""
    
    if dataset is None:
        return {}

    labels = []
    
    # Itera sobre o dataset não embaralhado (unbatch) para coletar os rótulos
    for _, batch_labels in dataset.unbatch().as_numpy_iterator():
        labels.append(np.argmax(batch_labels)) 
    
    train_labels = np.array(labels)
    
    # Calcular os pesos de classe
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    
    # >> CORREÇÃO APLICADA AQUI: Converte explicitamente o valor para float nativo
    class_weights = {i: float(weights[i]) for i in range(len(weights))}
    
    return class_weights

# O resto do seu código de inicialização:
CLASS_WEIGHTS = calculate_class_weights(train_dataset_init)
print("\n--- Pesos de Classe Calculados (class_weight) ---")
print(CLASS_WEIGHTS)