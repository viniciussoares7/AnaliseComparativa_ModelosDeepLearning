# tcc_models.py - Atualizado para usar tf.data.Dataset
import time
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG16, InceptionV3, ResNet50, EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import gc

# Importar configurações e constantes
from config import (
    STANDARD_IMAGE_SIZE, INCEPTION_IMAGE_SIZE, NUM_CLASSES, RESCALE_LAYER,
    CLASS_WEIGHTS
)

# Limpa a sessão anterior do Keras e coleta lixo para liberar memória
def clean_session():
    tf.keras.backend.clear_session()
    gc.collect()

# ===================================================================
# 3. MÉTODOS DE CONSTRUÇÃO DE MODELOS (Builders)
# ===================================================================

def add_classifier_head(base_model, num_classes=NUM_CLASSES, dropout_rate=0.5, custom_inputs=None):
    """ Adiciona o classificador (camadas densas) ao modelo base congelado. """
    
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Usa a entrada customizada, se fornecida (para incluir a camada de Rescaling)
    if custom_inputs is not None:
        model = Model(inputs=custom_inputs, outputs=predictions, name=base_model.name)
    else:
        model = Model(inputs=base_model.input, outputs=predictions, name=base_model.name)

    return model

def create_model_with_rescaling(base_model_class, input_shape, model_name):
    """ Função helper para construir modelos de Transfer Learning com Rescaling. """
    inputs = Input(shape=input_shape)
    x = RESCALE_LAYER(inputs) # Adiciona a camada de normalização [0, 255] -> [0, 1]
    
    # Garante que o input_tensor seja o 'x' normalizado
    base_model = base_model_class(weights='imagenet', include_top=False, input_tensor=x)
    return add_classifier_head(base_model, custom_inputs=inputs)

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
    model.add(RESCALE_LAYER)
    
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
# 4. MÉTODOS DE TREINAMENTO E AVALIAÇÃO
# ===================================================================

def get_callbacks(model_name):
    """ Retorna a lista de callbacks otimizados. """
    return [
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-6),
        ModelCheckpoint(f'{model_name}_best_model.h5', monitor='val_loss', save_best_only=True)
    ]

def train_model(model, train_ds, val_ds, class_weights, model_name, epochs_transfer=10, epochs_finetune=15):
    """
    Função principal para treinar o modelo em duas fases (Transfer Learning e Fine-Tuning) 
    usando tf.data.Dataset.
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

        for layer in model.layers[:unfreeze_from]:
            layer.trainable = False
        for layer in model.layers[unfreeze_from:]:
            layer.trainable = True

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
        print(f"\n--- {model_name}: Treinamento Único (Total de {epochs_transfer + epochs_finetune} épocas) ---")
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        start_time_cnn = time.time()
        model.fit(
            train_ds, epochs=(epochs_transfer + epochs_finetune), validation_data=val_ds,
            class_weight=class_weights, callbacks=callbacks, verbose=1
        )
        total_training_time = (time.time() - start_time_cnn)
        
    return model, total_training_time

def evaluate_model(model, test_ds, model_name, total_training_time,class_names):
    """
    Avalia o modelo treinado, gera métricas e o relatório de classificação usando tf.data.Dataset.
    """
    
    print(f"\n📊 Avaliando {model_name} no Conjunto de Teste...")
    metrics_zeroed = {k: 0 for k in ['Acurácia Teste (%)', 'Precision (Weighted Avg)', 'Recall (Weighted Avg)', 'F1-Score (Weighted Avg)', 'Tempo de Treinamento (s)', 'Tamanho do Modelo (MB)']}

    try:
        model.load_weights(f'{model_name}_best_model.h5')
    except Exception as e:
        print(f"❌ Erro Crítico ao carregar pesos de {model_name}: {e}. Retornando métricas zeradas.")
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

    # Truncar o predito se o dataset for menor que o esperado (pode acontecer com o último batch)
    if len(predicted_classes) != len(true_classes):
        print("⚠️ Aviso: Os rótulos previstos e reais têm tamanhos diferentes. Ajustando para o menor tamanho.")
        min_len = min(len(predicted_classes), len(true_classes))
        predicted_classes = predicted_classes[:min_len]
        true_classes = true_classes[:min_len]


    # Relatório de Classificação
    report_dict = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True, zero_division=0)

    # Obter o tamanho do modelo salvo em MB
    model_size_mb = os.path.getsize(f'{model_name}_best_model.h5') / (1024 * 1024)

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