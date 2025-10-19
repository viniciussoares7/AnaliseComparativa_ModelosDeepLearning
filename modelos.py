# C:\TCC\modelos.py - Com Cache/Save na pasta 'Modelos'

import time
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from keras import layers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import gc

# Importar configurações e constantes
from config import (
    STANDARD_IMAGE_SIZE, INCEPTION_IMAGE_SIZE, NUM_CLASSES, MODELS_DIR
)

# ===================================================================
# 1. FUNÇÕES AUXILIARES
# ===================================================================

def clean_session():
    """ Limpa a sessão anterior do Keras e coleta lixo para liberar memória. """
    tf.keras.backend.clear_session()
    gc.collect()

def add_classifier_head(base_model, num_classes=NUM_CLASSES, dropout_rate=0.5, custom_inputs=None):
    """ Adiciona o classificador (camadas densas) ao modelo base congelado. """
    
    # Congela o modelo base
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Constrói o modelo completo (Input -> Rescaling -> Base Model -> Head)
    if custom_inputs is not None:
        model = Model(inputs=custom_inputs, outputs=predictions, name=base_model.name)
    else:
        model = Model(inputs=base_model.input, outputs=predictions, name=base_model.name)

    return model

def create_model_with_rescaling(base_model_class, input_shape, model_name):
    """ 
    Função helper para construir modelos de Transfer Learning.
    Cria uma nova instância de Rescaling com nome único para cada modelo.
    """
    inputs = Input(shape=input_shape)

    # CORREÇÃO: Cria uma NOVA instância de Rescaling com nome único para este modelo
    rescale_layer_unique = layers.Rescaling(
        scale=1.0/255, 
        name=f"rescaling_{model_name}" # Nome Único (ex: rescaling_EfficientNetB0)
    )
    x = rescale_layer_unique(inputs)
    
    # Garante que o input_tensor seja o 'x' normalizado
    base_model = base_model_class(weights='imagenet', include_top=False, input_tensor=x)
    return add_classifier_head(base_model, custom_inputs=inputs)

# ===================================================================
# 2. MÉTODOS DE CONSTRUÇÃO DE MODELOS (Builders)
# ===================================================================

def build_vgg16_model(input_shape=STANDARD_IMAGE_SIZE + (3,)):
    """ Constrói o modelo VGG16 com Transfer Learning. """
    return create_model_with_rescaling(VGG16, input_shape, 'VGG16')

def build_inceptionv3_model(input_shape=INCEPTION_IMAGE_SIZE + (3,)):
    """ Constrói o modelo InceptionV3 (exige 299x299). """
    return create_model_with_rescaling(InceptionV3, input_shape, 'InceptionV3')

def build_resnet50_model(input_shape=STANDARD_IMAGE_SIZE + (3,)):
    """ Constrói o modelo ResNet50 com Transfer Learning. """
    return create_model_with_rescaling(ResNet50, input_shape, 'ResNet50')

def build_efficientnetb0_model(input_shape=STANDARD_IMAGE_SIZE + (3,)):
    """ Constrói o modelo EfficientNetB0 com Transfer Learning. """
    return create_model_with_rescaling(EfficientNetB0, input_shape, 'EfficientNetB0')

def build_custom_cnn(input_shape=STANDARD_IMAGE_SIZE + (3,)):
    """ Constrói a Rede Neural Convolucional Personalizada (CNN). """
    model = Sequential(name='Custom_CNN')

    # Adicionar a camada de normalização primeiro!
    model.add(Input(shape=input_shape))
    # CORREÇÃO: Cria uma NOVA instância de Rescaling com nome único
    model.add(layers.Rescaling(1./255, name='rescaling_custom_cnn'))
    
    # Bloco 1
    model.add(Conv2D(32, (3, 3), activation='relu'))
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
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model

# ===================================================================
# 3. MÉTODOS DE TREINAMENTO, CACHE E AVALIAÇÃO
# ===================================================================

def get_callbacks(model_name):
    """ Retorna a lista de callbacks otimizados, salvando pesos na raiz temporariamente. """
    # O ModelCheckpoint salva apenas os pesos (h5), permitindo que a função principal carregue.
    return [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6),
        ModelCheckpoint(f'{model_name}_best_model.h5', monitor='val_loss', save_best_only=True)
    ]

def train_model(model, train_ds, val_ds, class_weights, model_name, epochs_transfer=10, epochs_finetune=15):
    """
    Função principal para treinar o modelo em duas fases ou treinamento único (CNN).
    """
    
    callbacks = get_callbacks(model_name)
    total_training_time = 0.0

    # ----------------------------------------------------
    # FASE 1: Transfer Learning (Treinar Camadas Finais)
    # ----------------------------------------------------
    if epochs_transfer > 0 and model_name != 'Custom_CNN':
        print(f"\n--- {model_name}: FASE 1: Transfer Learning (Camadas Congeladas) ---")
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        start_time_transfer = time.time()
        model.fit(
            train_ds, epochs=epochs_transfer, validation_data=val_ds,
            class_weight=class_weights, callbacks=callbacks, verbose=1
        )
        total_training_time += (time.time() - start_time_transfer)
        
        try:
            model.load_weights(f'{model_name}_best_model.h5')
        except:
            print(f"⚠️ Aviso: Não foi possível carregar os pesos da Fase 1 para {model_name}. Continuando...")


    # ----------------------------------------------------
    # FASE 2: Fine-Tuning (Descongelar Camadas) ou CNN (Treinamento Único)
    # ----------------------------------------------------
    
    if model_name != 'Custom_CNN' and epochs_finetune > 0:
        print(f"\n--- {model_name}: FASE 2: Fine-Tuning (Descongelando Camadas) ---")
        
        unfreeze_map = {'VGG16': 16, 'InceptionV3': 249, 'ResNet50': 143, 'EfficientNetB0': 180}
        unfreeze_from = unfreeze_map.get(model_name, len(model.layers))

        # Descongela as camadas do modelo base para Fine-Tuning
        for layer in model.layers[1:]: # Ignora a camada de Rescaling
            layer.trainable = True
        for layer in model.layers[:unfreeze_from]:
            layer.trainable = False
        
        # Recompilação com LR menor
        model.compile(
            optimizer=Adam(learning_rate=1e-5), # LR 10x menor para Fine-Tuning
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        start_time_finetune = time.time()
        model.fit(
            train_ds, epochs=epochs_finetune, validation_data=val_ds,
            class_weight=class_weights, callbacks=callbacks, verbose=1
        )
        total_training_time += (time.time() - start_time_finetune)

    elif model_name == 'Custom_CNN':
        # Treinamento único para CNN customizada
        epochs = epochs_transfer + epochs_finetune
        print(f"\n--- {model_name}: Treinamento Único (Total de {epochs} épocas) ---")
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        start_time_cnn = time.time()
        model.fit(
            train_ds, epochs=epochs, validation_data=val_ds,
            class_weight=class_weights, callbacks=callbacks, verbose=1
        )
        total_training_time = (time.time() - start_time_cnn)
        
    return model, total_training_time

def _run_training(model_name, model_builder_func, train_ds, val_ds, class_weights, model_params, final_model_path):
    """ Função auxiliar para construir, treinar e salvar o modelo. """
    
    clean_session() # Garante um ambiente limpo
    
    # 1. Construir o modelo
    model = model_builder_func(**model_params)
    
    # 2. Treinar o modelo
    if model_name == 'Custom_CNN':
         trained_model, total_time = train_model(
            model, train_ds, val_ds, class_weights, model_name, epochs_transfer=0, epochs_finetune=25
        )
    else:
        trained_model, total_time = train_model(
            model, train_ds, val_ds, class_weights, model_name, epochs_transfer=10, epochs_finetune=15
        )

    # 3. Carregar os melhores pesos salvos pelo ModelCheckpoint durante o treino
    best_weights_path = f'{model_name}_best_model.h5'
    try:
        trained_model.load_weights(best_weights_path)
    except Exception as e:
        print(f"⚠️ Aviso: Não foi possível carregar os melhores pesos do ModelCheckpoint para salvar em {model_name}. Usando o modelo final da FASE 2. Erro: {e}")

    # 4. Salvar o modelo final treinado no caminho de cache (Modelo Completo)
    print(f"💾 Salvando modelo final '{model_name}' em: {final_model_path}")
    trained_model.save(final_model_path)
    
    # 5. Opcional: Remover o arquivo de checkpoint temporário
    if os.path.exists(best_weights_path):
        os.remove(best_weights_path)
        
    return trained_model, total_time


def load_or_train_model(model_name, model_builder_func, train_ds, val_ds, class_weights, model_params):
    """
    Tenta carregar o modelo de cache. Se não existir, treina, salva na pasta MODELS_DIR e retorna.
    """
    
    # 1. Definir o caminho completo do modelo no diretório 'Modelos'
    model_dir = os.path.join(MODELS_DIR, model_name)
    model_path = os.path.join(model_dir, f'{model_name}_final_model.h5')
    
    # Garantir que a pasta de destino exista
    os.makedirs(model_dir, exist_ok=True)
    
    trained_model = None
    total_time = 0.0
    
    # 2. Verificar se o modelo já existe (cache)
    if os.path.exists(model_path):
        print(f"✅ Modelo '{model_name}' encontrado em cache. Carregando para avaliação...")
        
        try:
            # Carrega o modelo COMPLETO (arquitetura + pesos + otimizador)
            trained_model = tf.keras.models.load_model(model_path)
            # O tempo de treino é 0.0 se for carregado do cache
            
        except Exception as e:
            print(f"❌ Erro ao carregar o modelo em cache '{model_path}': {e}. Iniciando novo treino...")
            # Se falhar, limpa a sessão e tenta treinar
            clean_session() 
            trained_model, total_time = _run_training(model_name, model_builder_func, train_ds, val_ds, class_weights, model_params, model_path)
            
    else:
        print(f"🔥 Modelo '{model_name}' não encontrado. Iniciando Treinamento...")
        trained_model, total_time = _run_training(model_name, model_builder_func, train_ds, val_ds, class_weights, model_params, model_path)
        
    return trained_model, total_time


def evaluate_model(model, test_ds, model_name, total_training_time,class_names):
    """
    Avalia o modelo treinado, gera métricas e o relatório de classificação usando tf.data.Dataset.
    Nota: Se o modelo veio do cache, os pesos já estão nele. Se veio do treino, os melhores pesos 
    (salvos pelo ModelCheckpoint) foram carregados na função _run_training antes de salvar.
    """
    
    print(f"\n📊 Avaliando {model_name} no Conjunto de Teste...")
    metrics_zeroed = {k: 0 for k in ['Acurácia Teste (%)', 'Precision (Weighted Avg)', 'Recall (Weighted Avg)', 'F1-Score (Weighted Avg)', 'Tempo de Treinamento (s)', 'Tamanho do Modelo (MB)']}
    
    # Obtém o caminho do modelo salvo (dentro da pasta Modelos)
    model_dir = os.path.join(MODELS_DIR, model_name)
    model_path = os.path.join(model_dir, f'{model_name}_final_model.h5')

    # Se a avaliação ocorrer logo após o treino, o arquivo de checkpoint temporário
    # já foi removido, mas o modelo final está salvo em model_path.
    if not os.path.exists(model_path):
        print(f"❌ Erro: Arquivo de modelo final esperado em {model_path} não encontrado.")
        return metrics_zeroed

    # Predições
    predictions = model.predict(test_ds, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    # Obter rótulos reais do test_ds
    true_classes_list = []
    class_labels_dict = class_names
    
    for _, batch_labels in test_ds.unbatch().as_numpy_iterator():
        true_classes_list.append(np.argmax(batch_labels)) 
    
    true_classes = np.array(true_classes_list)
    class_labels = class_labels_dict

    # Truncar o predito se os tamanhos forem diferentes
    if len(predicted_classes) != len(true_classes):
        print("⚠️ Aviso: Os rótulos previstos e reais têm tamanhos diferentes. Ajustando para o menor tamanho.")
        min_len = min(len(predicted_classes), len(true_classes))
        predicted_classes = predicted_classes[:min_len]
        true_classes = true_classes[:min_len]


    # Relatório de Classificação
    report_dict = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True, zero_division=0)

    # Obter o tamanho do modelo salvo em MB (Agora do caminho de cache)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    print(f"\n⏱️ Tempo Total de Treino ({model_name}): {total_training_time:.2f} segundos")
    print("\nRelatório de classificação (Teste):\n", classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0))
    print(f"\nTamanho Final do Modelo ({model_name}): {model_size_mb:.2f} MB")

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