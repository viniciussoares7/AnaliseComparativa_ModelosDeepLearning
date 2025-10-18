# main.py
import pandas as pd
import tensorflow as tf
from config import (
    STANDARD_IMAGE_SIZE, INCEPTION_IMAGE_SIZE,
    train_generator, validation_generator, test_generator, CLASS_WEIGHTS
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

def set_generator_target_size(size):
    """Ajusta o target_size dos generators (requer que os generators estejam carregados)."""
    if train_generator and validation_generator and test_generator:
        train_generator.target_size = size
        validation_generator.target_size = size
        test_generator.target_size = size
    else:
        print("Erro: Os generators de dados não foram carregados corretamente (Verifique config.py)")
    
def main():
    
    # Executa a limpeza inicial e verifica se o ambiente está pronto
    clean_session()
    
    if not train_generator:
        print("EXECUÇÃO INTERROMPIDA: Verifique o caminho DATASET_BASE_PATH no config.py.")
        return

    model_names = ['VGG16', 'InceptionV3', 'ResNet50', 'EfficientNetB0', 'Custom_CNN']
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
            
        # 2. CONFIGURAR GENERATORS:
        set_generator_target_size(current_image_size)

        # 3. CONSTRUIR O MODELO:
        model = model_builder()

        # 4. TREINAR O MODELO (25 épocas total):
        if name == 'Custom_CNN':
            trained_model, total_time = train_model(
                model, train_generator, validation_generator, CLASS_WEIGHTS, name, epochs_transfer=0, epochs_finetune=25
            )
        else:
            trained_model, total_time = train_model(
                model, train_generator, validation_generator, CLASS_WEIGHTS, name, epochs_transfer=10, epochs_finetune=15
            )

        # 5. AVALIAR O MODELO E COLETAR RESULTADOS:
        results = evaluate_model(trained_model, test_generator, name, total_time)
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