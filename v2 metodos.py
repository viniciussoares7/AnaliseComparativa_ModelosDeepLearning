# @title 1. Configuração Inicial e Bibliotecas
import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import gc # Importa gc para limpeza de sessão

# Limpa a sessão anterior do Keras e coleta lixo para liberar memória
tf.keras.backend.clear_session()
gc.collect()

# Definições de caminhos e parâmetros
DATASET_BASE_PATH = "/Jute_Pest_Dataset"

# IMPORTANTE: InceptionV3 exige 299x299. Para Transfer Learning, o 224x224
# funciona para a maioria. Vamos manter 224 para os outros.
STANDARD_IMAGE_SIZE = (224, 224) 
INCEPTION_IMAGE_SIZE = (299, 299)
BATCH_SIZE = 32
NUM_CLASSES = 17 

TRAIN_DIR = os.path.join(DATASET_BASE_PATH, 'train')
VALIDATION_DIR = os.path.join(DATASET_BASE_PATH, 'val')
TEST_DIR = os.path.join(DATASET_BASE_PATH, 'test')

# Removido MODEL_MAP pois agora temos funções diretas

# @title 2. Geração de Dados (Data Augmentation) e Cálculo de Pesos de Classe
# --- Data Augmentation Agressivo (Para todas as classes, focando em robustez) ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,      # Aumento da rotação
    width_shift_range=0.2,  # Aumento dos shifts
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,         # Aumento do zoom
    horizontal_flip=True,
    brightness_range=[0.8, 1.2], # Variação de brilho
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255) # Apenas rescale para teste e validação

# Os generators são criados com o IMAGE_SIZE padrão. Serão ajustados antes do treinamento de cada modelo.
# O IMAGE_SIZE padrão aqui será usado para cálculo dos pesos de classe.
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=STANDARD_IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    seed=42
)

validation_generator = test_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=STANDARD_IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False, # Não embaralha para manter ordem de avaliação
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=STANDARD_IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False, # Não embaralha para o relatório de classificação
    seed=42
)

# --- Cálculo de Pesos de Classe (class_weight) ---
def calculate_class_weights(generator):
    train_labels = generator.classes
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = {i: weights[i] for i in range(len(weights))}
    return class_weights

CLASS_WEIGHTS = calculate_class_weights(train_generator)
print("\n--- Pesos de Classe Calculados (class_weight) ---")
print(CLASS_WEIGHTS)

# @title 3. Funções de Construção de Modelos (Métodos Específicos)

def add_classifier_head(base_model, num_classes, dropout_rate=0.5):
    """ Adiciona o classificador (camadas densas) ao modelo base congelado. """
    
    # 1. Congelar todas as camadas da Base (Transfer Learning)
    for layer in base_model.layers:
        layer.trainable = False
        
    # 2. Adicionar novas camadas densas para classificação
    x = base_model.output
    x = GlobalAveragePooling2D()(x) # Redução de dimensão
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x) # Aumento do Dropout para regularização
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions, name=base_model.name)
    return model

# ----------------------------------------------------
# A. Modelos de Transfer Learning
# ----------------------------------------------------

def build_vgg16_model(input_shape=STANDARD_IMAGE_SIZE + (3,), num_classes=NUM_CLASSES):
    """ Constrói o modelo VGG16 com Transfer Learning. """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    return add_classifier_head(base_model, num_classes)

def build_inceptionv3_model(input_shape=INCEPTION_IMAGE_SIZE + (3,), num_classes=NUM_CLASSES):
    """ Constrói o modelo InceptionV3 (exige 299x299). """
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    return add_classifier_head(base_model, num_classes)

def build_resnet50_model(input_shape=STANDARD_IMAGE_SIZE + (3,), num_classes=NUM_CLASSES):
    """ Constrói o modelo ResNet50 com Transfer Learning. """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    return add_classifier_head(base_model, num_classes)

def build_efficientnetb0_model(input_shape=STANDARD_IMAGE_SIZE + (3,), num_classes=NUM_CLASSES):
    """ Constrói o modelo EfficientNetB0 com Transfer Learning. """
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    return add_classifier_head(base_model, num_classes)


# ----------------------------------------------------
# B. Modelo Personalizado (CNN)
# ----------------------------------------------------

def build_custom_cnn(input_shape=STANDARD_IMAGE_SIZE + (3,), num_classes=NUM_CLASSES):
    """ Constrói a Rede Neural Convolucional Personalizada (CNN). """
    model = Sequential(name='Custom_CNN')

    # Bloco 1
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Bloco 2
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Bloco 3
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))

    # Classificador
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# @title 4. Função de Treinamento e Avaliação (Duas Fases) - MANTIDA
# A função de treinamento foi mantida, mas a lógica de 'unfreeze'
# (descongelamento) foi ajustada para ser mais robusta.

def train_and_evaluate_model(model, train_gen, val_gen, test_gen, class_weights, model_name, epochs_transfer=10, epochs_finetune=15):

    print(f"\n=======================================================")
    print(f"🚀 Iniciando Treinamento e Fine-Tuning de {model_name}...")
    print(f"=======================================================")

    # --- Callbacks Otimizados (Combate ao Overfitting) ---
    callbacks = [
        # Monitora val_loss (perda), que é mais sensível ao overfitting.
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        # Reduz LR se val_loss estagnar.
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6),
        # Salva o melhor modelo com base na menor val_loss
        ModelCheckpoint(f'{model_name}_best_model.h5', monitor='val_loss', save_best_only=True)
    ]

    # Ajustar generators para o tamanho correto do modelo
    if model_name == 'InceptionV3':
        target_size = INCEPTION_IMAGE_SIZE
    else:
        target_size = STANDARD_IMAGE_SIZE

    train_gen.target_size = target_size
    val_gen.target_size = target_size
    test_gen.target_size = target_size

    # ----------------------------------------------------
    # FASE 1: Transfer Learning (Treinar Camadas Finais)
    # ----------------------------------------------------

    print("\n--- FASE 1: Transfer Learning (Camadas Congeladas) ---")

    model.compile(
        optimizer=Adam(learning_rate=1e-4), # LR inicial mais baixo e estável
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    start_time_transfer = time.time()

    # O fit pode ser chamado com 0 épocas se for a CNN customizada
    if epochs_transfer > 0:
        model.fit(
            train_gen,
            epochs=epochs_transfer,
            validation_data=val_gen,
            class_weight=class_weights, # ESSENCIAL: Usando pesos de classe
            callbacks=callbacks
        )

    end_time_transfer = time.time()
    total_training_time = end_time_transfer - start_time_transfer

    # Se a FASE 1 ocorreu (Transfer Learning), carrega os melhores pesos para o Fine-Tuning
    if epochs_transfer > 0:
        try:
            model.load_weights(f'{model_name}_best_model.h5')
        except:
             print(f"⚠️ Aviso: Não foi possível carregar os pesos da Fase 1 para {model_name}. Continuando...")


    # ----------------------------------------------------
    # FASE 2: Fine-Tuning (Descongelar e Treinar com LR muito baixo)
    # ----------------------------------------------------

    if model_name not in ['Custom_CNN'] and epochs_finetune > 0:
        print("\n--- FASE 2: Fine-Tuning (Descongelando Camadas) ---")

        # DEFINIÇÃO DO PONTO DE DESCONGELAMENTO (Fine-Tuning)
        # Os índices baseados em nomes de camadas são mais robustos do que índices numéricos
        if model_name == 'VGG16':
            # Descongela o 'block5_conv1' e tudo a seguir (índice 16)
            unfreeze_from = 16
        elif model_name == 'InceptionV3':
            # Descongela o bloco 'mixed7' (por volta da camada 249)
            unfreeze_from = 249
        elif model_name == 'ResNet50':
            # Descongela o 'conv5_block3' e tudo a seguir (índice 143)
            unfreeze_from = 143
        elif model_name == 'EfficientNetB0':
            # Descongela o último bloco MBConv (a partir da camada 180)
            unfreeze_from = 180
        else:
            unfreeze_from = len(model.layers) # Não deve ocorrer

        # Implementar o descongelamento
        for layer in model.layers[:unfreeze_from]:
            layer.trainable = False
        for layer in model.layers[unfreeze_from:]:
            layer.trainable = True

        # Recompilar com um Learning Rate MUITO baixo
        model.compile(
            optimizer=Adam(learning_rate=1e-5), # LR 10x menor para Fine-Tuning
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        start_time_finetune = time.time()

        model.fit(
            train_gen,
            epochs=epochs_finetune,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks
        )

        end_time_finetune = time.time()
        total_training_time += (end_time_finetune - start_time_finetune) # Soma o tempo da fase 2

    elif model_name == 'Custom_CNN': # Treinamento único para CNN personalizada
        # Como FASE 1 foi pulada (epochs_transfer=0), aqui rodamos o total de épocas
        
        # Recompilar com um LR padrão/ajustado para CNN do zero
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        start_time_cnn = time.time()
        
        model.fit(
            train_gen,
            epochs=epochs_transfer + epochs_finetune, # Total de 25 épocas
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks
        )
        
        end_time_cnn = time.time()
        total_training_time = end_time_cnn - start_time_cnn

    # ----------------------------------------------------
    # AVALIAÇÃO FINAL E GERAÇÃO DE MÉTRICAS
    # ----------------------------------------------------

    print(f"\n📊 Avaliando {model_name} no Conjunto de Teste...")

    # Carregar os pesos que tiveram o melhor desempenho (salvos pelo ModelCheckpoint)
    try:
        model.load_weights(f'{model_name}_best_model.h5')
    except:
        print(f"❌ Erro Crítico: Arquivo de pesos '{model_name}_best_model.h5' não encontrado ou corrompido.")
        metrics = {
             k: 0 for k in ['Acurácia Teste (%)', 'Precision (Weighted Avg)', 'Recall (Weighted Avg)', 'F1-Score (Weighted Avg)', 'Tempo de Treinamento (s)', 'Tamanho do Modelo (MB)']
        }
        return metrics # Retorna métricas zeradas em caso de falha

    # Predições
    predictions = model.predict(test_gen, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    # Relatório de Classificação
    report_dict = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True, zero_division=0)

    # Obter o tamanho do modelo salvo em MB
    model_size_mb = os.path.getsize(f'{model_name}_best_model.h5') / (1024 * 1024)

    print(f"\n⏱️ Tempo Total de Treino ({model_name}): {total_training_time:.2f} segundos")
    print("\nRelatório de classificação (Teste):\n", classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0))
    print(f"\nTamanho Final do Modelo ({model_name}): {model_size_mb:.2f} MB")

    # Limpar a sessão para a próxima iteração
    tf.keras.backend.clear_session()

    # Retornar as métricas consolidadas
    metrics = {
        'Acurácia Teste (%)': report_dict['accuracy'] * 100,
        'Precision (Weighted Avg)': report_dict['weighted avg']['precision'],
        'Recall (Weighted Avg)': report_dict['weighted avg']['recall'],
        'F1-Score (Weighted Avg)': report_dict['weighted avg']['f1-score'],
        'Tempo de Treinamento (s)': total_training_time,
        'Tamanho do Modelo (MB)': model_size_mb
    }
    return metrics


# @title 5. Execução Principal dos Modelos e Consolidação Final
# Lista de modelos de Transfer Learning (apenas os nomes)
transfer_models = ['VGG16', 'InceptionV3', 'ResNet50', 'EfficientNetB0']
all_results = {}

# Mapeamento dos nomes dos modelos para suas funções de construção
BUILD_FUNCTION_MAP = {
    'VGG16': build_vgg16_model,
    'InceptionV3': build_inceptionv3_model,
    'ResNet50': build_resnet50_model,
    'EfficientNetB0': build_efficientnetb0_model,
    'Custom_CNN': build_custom_cnn
}

# --- Execução dos Modelos de Transfer Learning ---
for name in transfer_models:
    # 1. Definir o target size e obter o modelo
    if name == 'InceptionV3':
        # InceptionV3 usa (299, 299, 3)
        current_image_size = INCEPTION_IMAGE_SIZE
        model = BUILD_FUNCTION_MAP[name]()
    else:
        # VGG16, ResNet50, EfficientNetB0 usam (224, 224, 3)
        current_image_size = STANDARD_IMAGE_SIZE
        model = BUILD_FUNCTION_MAP[name]()

    # 2. Seta o target size correto para os generators (Obrigatorio!)
    train_generator.target_size = current_image_size
    validation_generator.target_size = current_image_size
    test_generator.target_size = current_image_size

    # 3. Treinar e avaliar (duas fases: transfer + fine-tuning)
    results = train_and_evaluate_model(model, train_generator, validation_generator, test_generator, CLASS_WEIGHTS, name, epochs_transfer=10, epochs_finetune=15)
    all_results[name] = results
    
    # 4. Limpar a sessão (já feito dentro da função train_and_evaluate_model)


# --- Execução da CNN Personalizada ---
name = 'Custom_CNN'
current_image_size = STANDARD_IMAGE_SIZE

# 1. Definir o modelo (usa 224x224)
model = BUILD_FUNCTION_MAP[name]()

# 2. Seta o target size correto para os generators
train_generator.target_size = current_image_size
validation_generator.target_size = current_image_size
test_generator.target_size = current_image_size

# 3. Treinar e avaliar (ciclo único de 25 épocas)
cnn_results = train_and_evaluate_model(model, train_generator, validation_generator, test_generator, CLASS_WEIGHTS, name, epochs_transfer=0, epochs_finetune=25)
all_results['CNN Personalizada'] = cnn_results

# --- Consolidação Final na Tabela 2 ---
final_df = pd.DataFrame.from_dict(all_results, orient='index')

print("\n=======================================================")
print("TABELA 2 - RESULTADOS CONSOLIDADOS (APÓS OTIMIZAÇÃO)")
print("=======================================================")
# Formata o DataFrame para o padrão do seu TCC
final_df = final_df.rename_axis('Algoritmos').reset_index()

# Exibe o resultado final com a formatação solicitada
print(final_df.to_markdown(floatfmt=".4f"))