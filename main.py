import os
import pandas as pd
import tensorflow as tf
from tabulate import tabulate
from config import (
    STANDARD_IMAGE_SIZE, INCEPTION_IMAGE_SIZE, BATCH_SIZE, CLASS_WEIGHTS, MODELS_DIR,
    create_datasets, PLOTS_DIR 
)
from modelos import (
    build_vgg16_model, build_inceptionv3_model, build_resnet50_model,
    build_efficientnetb0_model, build_custom_cnn, load_or_train_model, 
    evaluate_model, clean_session
)

# Mapeamento dos nomes dos modelos para suas funções de construção
BUILD_FUNCTION_MAP = {
    'VGG16': build_vgg16_model,
    'InceptionV3': build_inceptionv3_model,
    'ResNet50': build_resnet50_model,
    'EfficientNetB0': build_efficientnetb0_model,
    'Custom_CNN': build_custom_cnn
}

def main():
    # Executa a limpeza inicial e cria as pastas necessárias
    clean_session()
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Verifica se os pesos de classe foram calculados
    if not CLASS_WEIGHTS:
        print("EXECUÇÃO INTERROMPIDA: Falha no carregamento dos dados ou cálculo de pesos.")
        return

    # Imprime os pesos de classe calculados usando tabulate para melhor visualização
    print("\n--- Pesos de Classe Calculados (class_weight) ---")
    print(tabulate(CLASS_WEIGHTS.items(), headers=['Classe Index', 'Peso'], tablefmt="simple"))
    print("--------------------------------------------------")


    #model_names = ['VGG16', 'InceptionV3', 'ResNet50', 'EfficientNetB0', 'Custom_CNN']
    model_names = ['ResNet50','Custom_CNN']

    all_results = {}

    for name in model_names:
        print(f"\n=======================================================")
        print(f"INICIANDO O PROCESSO COMPLETO PARA: {name}")
        print(f"=======================================================")
        
        # 1. SETUP: Determinar o tamanho da imagem e builder
        if name == 'InceptionV3':
            current_image_size = INCEPTION_IMAGE_SIZE
        else:
            current_image_size = STANDARD_IMAGE_SIZE
            
        model_builder = BUILD_FUNCTION_MAP[name]
            
        # 2. CONFIGURAR DATASETS: Recriar os Datasets com o tamanho correto
        current_train_ds, current_val_ds, current_test_ds, class_names = create_datasets(current_image_size, BATCH_SIZE)
        
        if current_train_ds is None:
            print(f"Pulando {name} devido a erro no carregamento do dataset.")
            continue

        # 3. CARREGAR/TREINAR O MODELO (Lógica com Cache):
        input_shape = current_image_size + (3,) 
        
        # Parâmetros passados para a função de construção
        model_params = {'input_shape': input_shape}
        
        trained_model, total_time, history = load_or_train_model( 
            model_name=name,
            model_builder_func=model_builder,
            train_ds=current_train_ds,
            val_ds=current_val_ds,
            class_weights=CLASS_WEIGHTS,
            model_params=model_params
        )
        
        # 4. AVALIAR O MODELO E COLETAR RESULTADOS:
        if trained_model:
            results = evaluate_model(trained_model, current_test_ds, name, total_time, class_names, history) 
            all_results[name] = results
        
        # 5. LIMPEZA:
        clean_session()
        
    # --- Consolidação Final na Tabela 2 ---
    if all_results:
        final_df = pd.DataFrame.from_dict(all_results, orient='index')

        print("\n=======================================================")
        print("TABELA 2 - RESULTADOS CONSOLIDADOS (APÓS OTIMIZAÇÃO)")
        print("=======================================================")
        
        final_df = final_df.rename_axis('Algoritmos').reset_index()
        final_df.loc[final_df['Algoritmos'] == 'Custom_CNN', 'Algoritmos'] = 'CNN Personalizada'

        # Usa tabulate para melhor formatação (especialmente para os floats)
        print(tabulate(
            final_df, 
            headers='keys', 
            tablefmt="pipe",
            showindex=False,
            numalign="decimal", # Ajuda a garantir que todos os números sejam tratados como tal
            floatfmt={
                None: ".2f", # <-- Adicionado formato padrão para qualquer float não listado
                'Acurácia Teste (%)': ".0f",
                'Precision (Weighted Avg)': ".4f",
                  'Recall (Weighted Avg)': ".4f",
                  'F1-Score (Weighted Avg)': ".4f",
                  'Tempo de Treinamento (s)': ".2f",
                  'Tamanho do Modelo (MB)': ".2f"
                }
                ))
    else:
        print("\nNenhum modelo foi treinado ou avaliado com sucesso.")

if __name__ == "__main__":
    main()