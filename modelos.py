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
from tensorflow.keras.utils import get_file
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Importar configurações e constantes
from config import (
    STANDARD_IMAGE_SIZE, INCEPTION_IMAGE_SIZE, NUM_CLASSES, MODELS_DIR,PLOTS_DIR,CLASS_WEIGHTS,MODEL_CONFIGS
)

# ===================================================================
# 1. FUNÇÕES AUXILIARES
# ===================================================================

def clean_session():
    """ Limpa a sessão anterior do Keras e coleta lixo para liberar memória. """
    tf.keras.backend.clear_session()
    gc.collect()

def add_classifier_head(x_features_input, num_classes, dropout_rate=0.5):
    """ 
    Recebe o tensor de FEATURES (saída da última camada conv do modelo base) 
    e anexa a cabeça do classificador (Pooling e Camadas Densas).
    Retorna o tensor de predições.
    """
    
    # 1. Aplica o Pooling no mapa de features
    x = GlobalAveragePooling2D()(x_features_input)
    
    # 2. Camada Densa 1 (Hidden Layer)
    x = Dense(512, activation='relu')(x)
    x = Dropout(dropout_rate)(x)
    
    # 3. Camada de Saída (Predictions)
    predictions = Dense(num_classes, activation='softmax')(x) 
    
    # Agora, a função retorna APENAS o tensor de saída.
    return predictions

def create_model_with_rescaling(base_model_class, input_shape, model_name):
    
    inputs = Input(shape=input_shape)
    rescale_layer_unique = layers.Rescaling(scale=1.0/255, name=f"rescaling_{model_name}")
    x_normalized = rescale_layer_unique(inputs) 
    
    # 1. Constrói o modelo base (a espinha dorsal).
    base_model = base_model_class(
        weights=None,
        include_top=False, 
        input_tensor=x_normalized, 
        input_shape=input_shape
    )
    
    # 2. CAPTURA A SAÍDA do modelo base. Este é o tensor de FEATURES.
    x_features = base_model.output 
    
    # 3. Adiciona a cabeça classificadora usando o tensor de FEATURES.
    predictions = add_classifier_head(x_features, num_classes=NUM_CLASSES)
    
    # 4. Define e retorna o modelo final.
    model = Model(inputs=inputs, outputs=predictions, name=model_name) 
    
    return model

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
    """ Constrói a Rede Neural Convolucional Personalizada (CNN) otimizada. """

    model = Sequential(name='Custom_CNN')

    # Adicionar a camada de normalização primeiro!
    model.add(Input(shape=input_shape))
    model.add(layers.Rescaling(1./255, name='rescaling_custom_cnn'))
    
    # Bloco 1
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Bloco 2
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Bloco 3
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    # Bloco 4 (NOVO)
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Classificador
    model.add(Flatten())
    
    # Camada Densa 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    # Camada Densa 2 (NOVO)
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    
    # Saída
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model

# ===================================================================
# 3. MÉTODOS DE TREINAMENTO, CACHE E AVALIAÇÃO
# ===================================================================

def get_callbacks(model_name,patience=10):
    """ Retorna a lista de callbacks otimizados, salvando APENAS pesos (H5). """
    filepath = f'{model_name}_best_model.weights.h5'

    return [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=int(patience/2), min_lr=1e-6),
        ModelCheckpoint(
            filepath, 
            monitor='val_loss', 
            save_best_only=True,
            save_weights_only=True, 
        )
    ]

def train_model(model, train_ds, val_ds, class_weights, model_name):
    
    start_time = time.time()
    
    # 1. Obter a configuração específica do modelo
    config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['VGG16']) 

    epochs_transfer = config['epochs_transfer']
    epochs_finetune = config['epochs_finetune']
    initial_lr = config['initial_lr']
    patience = config['patience']
    unfreeze_layers_count = config['unfreeze_layers_count']
    
    # Obter os Callbacks com a paciência específica
    callbacks_list = get_callbacks(model_name, patience=patience)

    history = None

    # --- FASE 1: Transfer Learning (Camadas Congeladas) ---
    if epochs_transfer > 0:
        print(f"\n--- {model_name}: FASE 1: Transfer Learning (Camadas Congeladas) ---")
        
        # Congelar todas as camadas do modelo base
        model.trainable = True 
        for layer in model.layers[:-len(model.layers[-1].weights)]:
            layer.trainable = False 

        # Compilação da Fase 1 (Learning Rate Inicial)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        # Treinamento da Fase 1
        history = model.fit(
            train_ds,
            epochs=epochs_transfer,
            validation_data=val_ds,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1
        )
        
    # --- FASE 2: Fine-Tuning (Descongelando Camadas) ---
    if epochs_finetune > 0:
        print(f"\n--- {model_name}: FASE 2: Fine-Tuning (Descongelando Camadas) ---")
        
        # Descongelar as camadas
        if model_name != 'Custom_CNN':
            

            num_layers = len(model.layers)
            
            # 1. Garante que todas as camadas do modelo base estejam treináveis, 
            # e a lógica abaixo define quem fica congelado ou não.
            for layer in model.layers:
                layer.trainable = True

            # 2. Aplica a lógica de congelamento/descongelamento:
            if unfreeze_layers_count > 0:
                # Descongelar N camadas do final (VGG16 padrão)
                layers_unfrozen = unfreeze_layers_count
                # As camadas já estão trainables=True, esta lógica é mais simples
            
            elif unfreeze_layers_count < 0:
                # Descongelar tudo, EXCETO as N primeiras camadas (ResNet50/EfficientNet)
                layers_to_keep_frozen = abs(unfreeze_layers_count)
                
                # Congela as N primeiras camadas (o "núcleo" da extração de features)
                for layer in model.layers[:layers_to_keep_frozen]:
                    layer.trainable = False
                
                # As camadas restantes (corpo principal do ResNet50) permanecem True
                layers_unfrozen = num_layers - layers_to_keep_frozen
            
            else:
                # Caso unfreeze_layers_count == 0 ou valor inválido
                layers_unfrozen = 0
                for layer in model.layers:
                    layer.trainable = False

            print(f"✅ Camadas descongeladas para Fine-Tuning: {layers_unfrozen} de {num_layers} camadas totais.")
        
        
        #Usa o LR de Fine-Tuning configurado (fine_tune_lr), se existir
        # Se não existir (como no Custom_CNN), usa o padrão (initial_lr * 0.1)
        finetune_lr = config.get('fine_tune_lr', initial_lr * 0.1) 
        
        # Compilação da Fase 2 (Novo Learning Rate mais baixo)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_lr),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy']
        )

        # Treinamento da Fase 2
        history_finetune = model.fit(
            train_ds,
            epochs=epochs_finetune,
            validation_data=val_ds,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1
        )
        
        # Mesclar históricos
        if history:
             for key in history_finetune.history.keys():
                 history.history[key].extend(history_finetune.history[key])
        else:
            history = history_finetune
            
    total_time = time.time() - start_time
    print(f"\n⏱️ Tempo Total de Treino ({model_name}): {total_time:.2f} segundos")

    return model, total_time, history


def _run_training(model_name, model_builder_func, train_ds, val_ds, class_weights, model_params, final_model_path):
    """ 
    Função auxiliar para construir, treinar e salvar o modelo.
    Inclui a correção de carregamento manual de pesos para EfficientNetB0.
    """
    
    clean_session() 
    
    # 1. Construir o modelo
    model = model_builder_func(**model_params)

    weights_path = None # Inicializa weights_path
    
    # 1.1. Carregamento manual para EfficientNetB0 e outros com pesos ImageNet
    if model_name in ['EfficientNetB0', 'VGG16', 'InceptionV3', 'ResNet50']:
        print(f"🔧 Aplicando correção de canal: Carregando pesos {model_name} ImageNet manualmente...")
        
        # Mapeamento para obter os pesos notop corretos
        WEIGHTS_MAP = {
            'EfficientNetB0': 'https://storage.googleapis.com/keras-applications/efficientnetb0_notop.h5',
            'VGG16': 'https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
            'InceptionV3': 'https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
            'ResNet50': 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        }
        
        weights_url = WEIGHTS_MAP.get(model_name)
        weights_filename = os.path.basename(weights_url)
        
        weights_path = get_file(
            weights_filename,
            weights_url,
            cache_subdir='models',
        )
        
        # Carrega os pesos ImageNet no modelo recém-criado 
        try:
            model.load_weights(weights_path, by_name=True, skip_mismatch=True)
            print("✅ Pesos ImageNet carregados com sucesso (manual).")
        except Exception as e:
             print(f"❌ Erro ao carregar pesos ImageNet para {model_name}: {e}")

    # 2. Treinar o modelo
    trained_model, total_time, history = train_model(
        model, 
        train_ds, 
        val_ds, 
        class_weights, 
        model_name, 
    )

    # 3. Carregar os melhores pesos salvos pelo ModelCheckpoint durante o treino
    best_weights_path = f'{model_name}_best_model.weights.h5'

    # 4. Salvar o modelo final treinado no caminho de cache
    print(f"💾 Salvando modelo final '{model_name}' em: {final_model_path}")

    if model_name in ['EfficientNetB0', 'VGG16', 'ResNet50', 'InceptionV3']:
        trained_model.save_weights(final_model_path)
        print(f"     (Apenas pesos salvos para modelos Transfer Learning.)")
    else:
        # Salva o modelo COMPLETO (arquitetura + pesos) para Custom_CNN
        trained_model.save(final_model_path)
    
    # 5. Remover o arquivo de checkpoint temporário
    if os.path.exists(best_weights_path):
        os.remove(best_weights_path)
        
    return trained_model, total_time, history


def load_or_train_model(model_name, model_builder_func, train_ds, val_ds, class_weights, model_params):
    """
    Tenta carregar o modelo de cache. Se não existir, treina, salva na pasta MODELS_DIR e retorna.
    """
    
    model_dir = os.path.join(MODELS_DIR, model_name)
    model_path = os.path.join(model_dir, f'{model_name}_final_model.h5')
    
    os.makedirs(model_dir, exist_ok=True)
    
    trained_model = None
    total_time = 0.0
    history = None
    
    # 2. Verificar se o modelo já existe (cache)
    if os.path.exists(model_path):
        print(f"✅ Modelo '{model_name}' encontrado em cache. Carregando para avaliação...")
        
        try:
            if model_name in ['EfficientNetB0', 'VGG16', 'ResNet50', 'InceptionV3']:
                # 🚨 Carregamento Pontual: Reconstrua e carregue apenas os pesos.
                clean_session() 
                trained_model = model_builder_func(**model_params)
                
                # Tenta carregar os pesos, se falhar o try/except cai no re-treino
                trained_model.load_weights(model_path)
                config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS['VGG16'])
                finetune_lr = config.get('fine_tune_lr', config['initial_lr'] * 0.1) 
                trained_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=finetune_lr),
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['accuracy']
                )

            else:
                # Carrega o modelo COMPLETO (Custom_CNN)
                trained_model = tf.keras.models.load_model(model_path)
            
        except Exception as e:
            # ERRO DE ARQUITETURA/PESOS/SERIALIZAÇÃO - Forçar novo treino
            print(f"❌ Erro ao carregar o modelo em cache '{model_path}': {e}. Iniciando novo treino...")
            clean_session() 
            trained_model, total_time, history = _run_training(model_name, model_builder_func, train_ds, val_ds, class_weights, model_params, model_path)
            
    else:
        print(f"🔥 Modelo '{model_name}' não encontrado. Iniciando Treinamento...")
        trained_model, total_time, history = _run_training(model_name, model_builder_func, train_ds, val_ds, class_weights, model_params, model_path)
        
    return trained_model, total_time, history


def evaluate_model(model, test_ds, model_name, total_training_time,class_names,history=None):
    """
    Avalia o modelo treinado, gera métricas e o relatório de classificação usando tf.data.Dataset.
    """
    
    print(f"\n📊 Avaliando {model_name} no Conjunto de Teste...")
    metrics_zeroed = {k: 0 for k in ['Acurácia Teste (%)', 'Precision (Weighted Avg)', 'Recall (Weighted Avg)', 'F1-Score (Weighted Avg)', 'Tempo de Treinamento (s)', 'Tamanho do Modelo (MB)']}
    
    model_dir = os.path.join(MODELS_DIR, model_name)
    model_path = os.path.join(model_dir, f'{model_name}_final_model.h5')

    if not os.path.exists(model_path):
        print(f"❌ Erro: Arquivo de modelo final esperado em {model_path} não encontrado.")
        return metrics_zeroed

    # Predições
    predictions = model.predict(test_ds, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)

    # Obter rótulos reais do test_ds
    true_classes_list = []
    
    for _, batch_labels in test_ds.unbatch().as_numpy_iterator():
        true_classes_list.append(np.argmax(batch_labels)) 
    
    true_classes = np.array(true_classes_list)
    class_labels = class_names

    # Truncar o predito se os tamanhos forem diferentes (caso raro)
    if len(predicted_classes) != len(true_classes):
        print("⚠️ Aviso: Os rótulos previstos e reais têm tamanhos diferentes. Ajustando para o menor tamanho.")
        min_len = min(len(predicted_classes), len(true_classes))
        predicted_classes = predicted_classes[:min_len]
        true_classes = true_classes[:min_len]

    # Relatório de Classificação
    report_dict = classification_report(true_classes, predicted_classes, target_names=class_labels, output_dict=True, zero_division=0)

    # Obter o tamanho do modelo salvo em MB
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

    # A. Salvar Matriz de Confusão
    plot_and_save_confusion_matrix(true_classes, predicted_classes, class_names, model_name)
    
    # B. Salvar Gráficos de Histórico (se o modelo foi treinado, não carregado do cache)
    if history is not None:
        plot_and_save_history_metrics(history, model_name)

    return metrics

# ===================================================================
# 4. FUNÇÕES DE PLOTAGEM E GERAÇÃO DE GRÁFICOS
# ===================================================================

def plot_and_save_history_metrics(history, model_name):
    """Gera e salva gráficos de Acurácia e Perda (Loss) por época."""
    
    plot_dir = os.path.join(PLOTS_DIR, model_name)
    os.makedirs(plot_dir, exist_ok=True)
    
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    # --- Gráfico de Acurácia ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.history['accuracy'], label='Acurácia de Treinamento')
    plt.plot(epochs, history.history['val_accuracy'], label='Acurácia de Validação')
    plt.title(f'Acurácia de Treinamento e Validação - {model_name}')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{model_name}_Acuracia_Historico.png'))
    plt.close()

    # --- Gráfico de Perda (Loss) ---
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history.history['loss'], label='Perda de Treinamento')
    plt.plot(epochs, history.history['val_loss'], label='Perda de Validação')
    plt.title(f'Perda de Treinamento e Validação - {model_name}')
    plt.xlabel('Época')
    plt.ylabel('Perda (Loss)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{model_name}_Perda_Historico.png'))
    plt.close()
    print(f"🖼️ Gráficos de Histórico salvos em: {plot_dir}")


def plot_and_save_confusion_matrix(true_classes, predicted_classes, class_names, model_name):
    """Gera e salva a Matriz de Confusão."""
    
    plot_dir = os.path.join(PLOTS_DIR, model_name)
    os.makedirs(plot_dir, exist_ok=True)

    cm = confusion_matrix(true_classes, predicted_classes)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Predito')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{model_name}_Matriz_Confusao.png'))
    plt.close()
    print(f"🖼️ Matriz de Confusão salva em: {plot_dir}")