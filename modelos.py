# config.py
import os
import numpy as np
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ===================================================================
# 1. CONSTANTES E CAMINHOS
# ===================================================================

# Variável que deve ser ajustada pelo usuário
DATASET_BASE_PATH = "/Jute_Pest_Dataset"

# Dimensões e Parâmetros
STANDARD_IMAGE_SIZE = (224, 224) 
INCEPTION_IMAGE_SIZE = (299, 299)
BATCH_SIZE = 32
NUM_CLASSES = 17 

# Caminhos para as subpastas
TRAIN_DIR = os.path.join(DATASET_BASE_PATH, 'train')
VALIDATION_DIR = os.path.join(DATASET_BASE_PATH, 'val')
TEST_DIR = os.path.join(DATASET_BASE_PATH, 'test')

# ===================================================================
# 2. DATA AUGMENTATION E GERADORES
# ===================================================================

# Data Augmentation Agressivo
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=30, width_shift_range=0.2, 
    height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, 
    horizontal_flip=True, brightness_range=[0.8, 1.2], fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Geração inicial dos generators (usando STANDARD_IMAGE_SIZE para cálculo de pesos)
try:
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=STANDARD_IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', seed=42
    )
    validation_generator = test_datagen.flow_from_directory(
        VALIDATION_DIR, target_size=STANDARD_IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False, seed=42
    )
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR, target_size=STANDARD_IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False, seed=42
    )
    
    # ===================================================================
    # 3. PESOS DE CLASSE
    # ===================================================================
    
    def calculate_class_weights(generator):
        """Calcula pesos de classe para mitigar o desbalanceamento."""
        train_labels = generator.classes
        weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels),
            y=train_labels
        )
        class_weights = {i: weights[i] for i in range(len(weights))}
        return class_weights

    CLASS_WEIGHTS = calculate_class_weights(train_generator)
    print("--- Pesos de Classe Calculados (class_weight) ---")
    print(CLASS_WEIGHTS)

except Exception as e:
    print(f"ERRO: Não foi possível carregar os datasets. Verifique DATASET_BASE_PATH e a estrutura de pastas: {e}")
    # Cria objetos vazios para evitar erros de importação na próxima etapa
    train_generator, validation_generator, test_generator = None, None, None
    CLASS_WEIGHTS = {}