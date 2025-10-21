# 📄 README: Classificação de Imagens com Transfer Learning e CNN Customizada

Este projeto implementa e compara quatro modelos de *Transfer Learning* (EfficientNetB0, VGG16, InceptionV3, ResNet50) e uma Rede Neural Convolucional (CNN) Customizada Otimizada para classificação de **17 classes** de pragas e doenças em imagens.

## 💻 1. Pré-requisitos e Instalação

O ambiente foi desenvolvido com foco em aceleração por GPU (CUDA/cuDNN), mas pode ser executado em CPU.

### A. Dependências do Sistema (Aceleração GPU - Recomendado)

Para usar a aceleração com TensorFlow, você deve ter os seguintes componentes instalados (verifique a documentação oficial do TensorFlow para as versões exatas de compatibilidade):

1.  **Driver NVIDIA:** Versão mais recente para sua placa de vídeo.
2.  **CUDA Toolkit:** Geralmente **CUDA 11.2** ou superior (compatível com TensorFlow 2.x).
3.  **cuDNN:** Versão correspondente ao CUDA Toolkit instalado.

### B. Instalação do Ambiente Python

Recomendamos o uso de um ambiente virtual (`venv` ou `conda`) para isolar as dependências.

1.  **Criação do Ambiente Virtual (exemplo com venv):**
    ```bash
    python -m venv cnn_env
    source cnn_env/bin/activate  # Linux/macOS
    cnn_env\Scripts\activate     # Windows
    ```

2.  **Instalação das Bibliotecas:**
    Salve o conteúdo abaixo em um arquivo chamado `requirements.txt` e use o `pip` para instalar:

    **Conteúdo de `requirements.txt`:**
    ```
    tensorflow
    numpy
    scikit-learn
    matplotlib
    seaborn
    keras
    ```

    **Comando de Instalação:**
    ```bash
    pip install -r requirements.txt
    ```

## 📂 2. Estrutura do Projeto

A execução correta do pipeline depende da seguinte estrutura de pastas:

. ├── src/ │ ├── main.py # (Script principal de execução) │ ├── modelos.py # (Onde estão as arquiteturas e a lógica de treino) │ ├── config.py # (Contém MODEL_CONFIGS, NUM_CLASSES, paths, etc.) │ └── data_loader.py # (Lógica de carregamento de datasets) ├── Dados/ │ ├── Treino/ # (Imagens de Treinamento) │ ├── Validacao/ # (Imagens de Validação) │ └── Teste/ # (Imagens de Teste) ├── Modelos/ # -> Pasta de cache para modelos e pesos (.h5) └── Resultados/ └── Graficos/ # -> Saída: Matrizes de Confusão e Históricos de Treino

## ⚙️ 3. Configuração e Execução

### Passo 3.1: Configurar a Lógica de Treino (`modelos.py`)

O arquivo `modelos.py` foi atualizado para otimizar o **ResNet50** e aprimorar o **Custom_CNN**. Certifique-se de que essas alterações estão em vigor.

### Passo 3.2: Ajustar Hiperparâmetros (`config.py`)

Verifique se a configuração do ResNet50 no seu dicionário `MODEL_CONFIGS` está com os valores otimizados para evitar o *overfitting* anterior:

```python
MODEL_CONFIGS = {
    # ...
    'ResNet50': {
        'initial_lr': 1e-4,
        'epochs_transfer': 10,
        
        # Parâmetros Otimizados
        'fine_tune_lr': 5e-6,        # LR reduzido para estabilidade
        'epochs_finetune': 30,       # Mais épocas para convergência lenta
        'patience': 10,              # Mais paciência para callbacks
        'unfreeze_layers_count': -20 # Descongelar tudo, exceto as 20 primeiras
    },
    # ...
}
```

### Passo 3.3: Iniciar o Treinamento

Execute o script principal do seu projeto:

```bash
python src/main.py
```

### Ordem de Execução e Saída:

O script executará ou carregará em cache cada modelo. Para os modelos que não estiverem em cache (como o ResNet50 e o novo Custom_CNN):

1.  **Fase de Treinamento:** Os logs de época e a detecção de GPU aparecerão no console.
2.  **Salvamento:** O melhor modelo será salvo em `Modelos/<Nome_Modelo>/<Nome_Modelo>_final_model.h5`.
3.  **Avaliação:** O relatório de classificação e a acurácia final (Teste) serão impressos no console.
4.  **Artefatos:** Os gráficos de histórico (`<Modelo>_Acuracia_Historico.png`, `<Modelo>_Perda_Historico.png`) e a matriz de confusão (`<Modelo>_Matriz_Confusao.png`) serão salvos em `Resultados/Graficos/<Nome_Modelo>/`.