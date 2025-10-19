# main.py - Atualizado para usar tf.data.Dataset
import pandas as pd
import tensorflow as tf
from config import (
    STANDARD_IMAGE_SIZE, INCEPTION_IMAGE_SIZE, BATCH_SIZE, CLASS_WEIGHTS,
    create_datasets
)
from modelos import (
    build_vgg16_model, build_inceptionv3_model, build_resnet50_model,
    build_efficientnetb0_model, build_custom_cnn, train_model, evaluate_model,
    clean_session
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

    # Executa a limpeza inicial
    clean_session()
    
    # Verifica se os pesos de classe foram calculados (indicando que os dados foram carregados)
    if not CLASS_WEIGHTS:
        print("EXECUÇÃO INTERROMPIDA: Falha no carregamento dos dados ou cálculo de pesos.")
        return

    model_names = ['VGG16', 'InceptionV3', 'ResNet50', 'EfficientNetB0', 'Custom_CNN']
    all_results = {}

    for name in model_names:
        print(f"\n=======================================================")
        print(f"INICIANDO O PROCESSO COMPLETO PARA: {name}")
        print(f"=======================================================")
        
        # 1. SETUP: Determinar o tamanho da imagem
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

        # 3. CONSTRUIR O MODELO:
        input_shape = current_image_size + (3,)
        model = model_builder(input_shape=input_shape)

        # 4. TREINAR O MODELO (25 épocas total):
        if name == 'Custom_CNN':
            trained_model, total_time = train_model(
                model, current_train_ds, current_val_ds, CLASS_WEIGHTS, name, epochs_transfer=0, epochs_finetune=25
            )
        else:
            trained_model, total_time = train_model(
                model, current_train_ds, current_val_ds, CLASS_WEIGHTS, name, epochs_transfer=10, epochs_finetune=15
            )

        # 5. AVALIAR O MODELO E COLETAR RESULTADOS:
        results = evaluate_model(trained_model, current_test_ds, name, total_time, class_names)
        all_results[name] = results
        
        # 6. LIMPEZA:
        clean_session()
        
    # --- Consolidação Final na Tabela 2 ---
    final_df = pd.DataFrame.from_dict(all_results, orient='index')

    print("\n=======================================================")
    print("TABELA 2 - RESULTADOS CONSOLIDADOS (APÓS OTIMIZAÇÃO)")
    print("=======================================================")
    
    final_df = final_df.rename_axis('Algoritmos').reset_index()
    final_df.loc[final_df['Algoritmos'] == 'Custom_CNN', 'Algoritmos'] = 'CNN Personalizada'

    print(final_df.to_markdown(floatfmt=".4f"))

if __name__ == "__main__":
    main()