# =============================================================================
# Identificador: GO1061-Tutorial07-SolucaoCompleta
# Aula 10 — MLP: Redes Neurais Multicamadas
# Tutorial Passo a Passo: Reconhecimento de Dígitos Escritos à Mão
# =============================================================================
#
# PARTE 7 DE 7 — SOLUCAO COMPLETA
# ============================================================
#
# Este arquivo une todos os passos anteriores em um único pipeline.
# Use como referência rápida ou ponto de partida para seus projetos.
#
# PIPELINE COMPLETO:
#   1. Carregar o dataset MNIST
#   2. Preparar os dados (normalizar, reshape)
#   3. Construir o modelo MLP
#   4. Compilar (configurar o aprendizado)
#   5. Treinar (com EarlyStopping)
#   6. Avaliar no conjunto de teste
#   7. Fazer predições e visualizar resultados
#   8. Salvar o modelo treinado
#
# REFERENCIA RAPIDA:
#   Para detalhes de cada etapa, consulte os arquivos:
#   GO1055 → O Problema        GO1058 → Construindo a Rede
#   GO1056 → Entendendo Dados  GO1059 → Treinando
#   GO1057 → Preparando Dados  GO1060 → Avaliacao
# =============================================================================

# matplotlib no nível do módulo — sempre, sem exceção
import matplotlib
import matplotlib.pyplot as plt

# Detecta ambiente de execução (Jupyter Notebook vs terminal)
try:
    get_ipython()
    matplotlib.use("module://matplotlib_inline.backend_inline")
except NameError:
    pass

import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# =============================================================================
# CONFIGURAÇÕES GLOBAIS
# =============================================================================

# Semente aleatória para reprodutibilidade
# Com a mesma semente, os resultados serão idênticos em execuções diferentes
SEMENTE = 42
np.random.seed(SEMENTE)
tf.random.set_seed(SEMENTE)

# Hiperparâmetros — reunidos aqui para fácil ajuste
EPOCHS     = 25        # máximo de épocas (EarlyStopping pode parar antes)
BATCH_SIZE = 128       # imagens por atualização de pesos
VAL_SPLIT  = 0.1       # 10% do treino para validação
PATIENCE   = 5         # épocas sem melhora antes de parar
DROPOUT_1  = 0.3       # dropout na camada 1 (30%)
DROPOUT_2  = 0.2       # dropout na camada 2 (20%)
NEURONS_1  = 512       # neurônios na camada oculta 1
NEURONS_2  = 256       # neurônios na camada oculta 2

# Caminho para salvar o modelo treinado
MODELO_PATH = "mnist_mlp_model.keras"


# =============================================================================
# PASSO 1: CARREGAR OS DADOS
# =============================================================================

def carregar_dados():
    """
    Carrega o MNIST diretamente do Keras.
    Retorna os dados brutos, sem nenhuma transformação.
    """
    print("=" * 60)
    print("SOLUCAO COMPLETA — TUTORIAL AULA 10")
    print("Reconhecimento de Digitos Escritos a Mao (MNIST)")
    print("=" * 60)
    print()
    print("[1/8] Carregando dataset MNIST...")

    (x_treino, y_treino), (x_teste, y_teste) = keras.datasets.mnist.load_data()

    print(f"      Treino: {x_treino.shape} | dtype={x_treino.dtype}")
    print(f"      Teste:  {x_teste.shape}  | dtype={x_teste.dtype}")
    print(f"      Classes: {np.unique(y_treino)}  (digitos 0 a 9)")
    return x_treino, y_treino, x_teste, y_teste


# =============================================================================
# PASSO 2: PREPARAR OS DADOS
# =============================================================================

def preparar_dados(x_treino, x_teste):
    """
    Aplica as transformações necessárias:
    1. Normalização: 0-255 → 0.0-1.0  (estabiliza o treinamento)
    2. Reshape: não necessário — usamos Flatten no modelo

    Os labels (y) ficam como inteiros — usamos sparse_categorical_crossentropy.
    """
    print()
    print("[2/8] Preparando os dados...")

    # Normalização: divide por 255.0 para escalar para [0.0, 1.0]
    x_treino_norm = x_treino.astype("float32") / 255.0
    x_teste_norm  = x_teste.astype("float32")  / 255.0

    print(f"      Normalizados: min={x_treino_norm.min():.1f} max={x_treino_norm.max():.1f}")
    print(f"      Shape: {x_treino_norm.shape} (Flatten sera feito pelo modelo)")
    return x_treino_norm, x_teste_norm


# =============================================================================
# PASSO 3: CONSTRUIR O MODELO
# =============================================================================

def construir_modelo():
    """
    Arquitetura MLP com 2 camadas ocultas:
        [Flatten] → [Dense 512 ReLU] → [Dropout 30%]
                  → [Dense 256 ReLU] → [Dropout 20%]
                  → [Dense 10 Softmax]

    Parâmetros totais: 535.818 treináveis
    """
    print()
    print("[3/8] Construindo o modelo MLP...")

    model = keras.Sequential(
        [
            # Flatten: transforma a imagem 28x28 em um vetor de 784 números
            # O modelo recebe imagens brutas (28,28) e faz o achatamento internamente
            layers.Flatten(input_shape=(28, 28), name="flatten"),

            # Camada oculta 1: 512 neurônios com ReLU
            # Aprende combinações de pixels (traços, curvas, bordas)
            layers.Dense(NEURONS_1, activation="relu", name="dense_1"),

            # Dropout 30%: desliga aleatoriamente 30% dos neurônios no treino
            # Evita que a rede "decore" os dados (overfitting)
            layers.Dropout(DROPOUT_1, name="dropout_1"),

            # Camada oculta 2: 256 neurônios com ReLU
            # Combina as características da camada 1 em padrões mais abstratos
            layers.Dense(NEURONS_2, activation="relu", name="dense_2"),

            # Dropout 20%: menor que o anterior (camada menor, menos neurônios a desligar)
            layers.Dropout(DROPOUT_2, name="dropout_2"),

            # Camada de saída: 10 neurônios (um por dígito 0-9) com Softmax
            # Softmax converte saídas brutas em probabilidades que somam 1.0 (100%)
            layers.Dense(10, activation="softmax", name="saida"),
        ],
        name="MLP_MNIST"
    )

    print(f"      Parametros treinaveis: {model.count_params():,}")
    print()
    model.summary()
    return model


# =============================================================================
# PASSO 4: COMPILAR O MODELO
# =============================================================================

def compilar_modelo(model):
    """
    Configura o processo de aprendizado:
    - optimizer='adam': ajusta os pesos de forma adaptativa
    - loss='sparse_categorical_crossentropy': mede o erro para labels inteiros
    - metrics=['accuracy']: monitora a proporção de acertos
    """
    print()
    print("[4/8] Compilando o modelo...")

    model.compile(
        # Adam: combina RMSprop e Momentum — funciona bem na maioria dos casos
        optimizer="adam",
        # sparse_categorical: aceita labels como inteiros (0,1,...,9)
        # Em vez de one-hot encoding, o Keras faz isso internamente
        loss="sparse_categorical_crossentropy",
        # Accuracy: % de predições corretas — fácil de interpretar
        metrics=["accuracy"],
    )

    print("      optimizer: adam")
    print("      loss: sparse_categorical_crossentropy")
    print("      metrics: accuracy")
    return model


# =============================================================================
# PASSO 5: TREINAR O MODELO
# =============================================================================

def treinar_modelo(model, x_treino, y_treino):
    """
    Executa o treinamento com EarlyStopping e ReduceLROnPlateau.

    EarlyStopping: para o treino quando val_loss para de melhorar.
    ReduceLROnPlateau: reduz a taxa de aprendizado quando estagna.
    Ambos ajudam a evitar overfitting e economizam tempo.
    """
    print()
    print("[5/8] Treinando o modelo...")
    print(f"      epochs={EPOCHS}, batch_size={BATCH_SIZE}, val_split={VAL_SPLIT}")
    print()

    callbacks = [
        # Para o treino se val_loss não melhorar por PATIENCE épocas
        # restore_best_weights=True: volta para o melhor checkpoint
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduz a taxa de aprendizado à metade se val_loss estagnar por 3 épocas
        # Permite ajustes mais finos quando a rede está quase convergida
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    history = model.fit(
        x_treino,
        y_treino,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT,
        callbacks=callbacks,
        verbose=1,
    )

    print()
    epocas_reais = len(history.history["loss"])
    print(f"      Treinamento concluido em {epocas_reais} epocas!")
    return history


# =============================================================================
# PASSO 6: AVALIAR NO CONJUNTO DE TESTE
# =============================================================================

def avaliar_modelo(model, x_teste, y_teste):
    """
    Avaliação final no conjunto de teste (dados nunca vistos pelo modelo).
    Esta é a métrica "oficial" do nosso modelo.
    """
    print()
    print("[6/8] Avaliando no conjunto de teste...")

    loss, accuracy = model.evaluate(x_teste, y_teste, verbose=0)

    print(f"      Loss no teste:     {loss:.4f}")
    print(f"      Acuracia no teste: {accuracy:.4f} ({accuracy*100:.2f}%)")

    if accuracy >= 0.98:
        erros = int((1 - accuracy) * len(y_teste))
        print(f"      META ATINGIDA (>=98%)! Apenas {erros} erros em {len(y_teste):,} imagens.")
    else:
        print(f"      Abaixo da meta. Tente ajustar hiperparametros.")

    return loss, accuracy


# =============================================================================
# PASSO 7: PREDIÇÕES E ANÁLISE
# =============================================================================

def obter_predicoes(model, x_teste, y_teste):
    """
    Gera predições para todo o conjunto de teste.
    Retorna tanto as probabilidades quanto as classes preditas.
    """
    print()
    print("[7/8] Gerando predicoes...")

    # probabilidades: (10000, 10) — probabilidade de cada classe para cada imagem
    probabilidades = model.predict(x_teste, verbose=0)

    # predicoes: (10000,) — índice da classe com maior probabilidade
    predicoes = np.argmax(probabilidades, axis=1)

    acertos = np.sum(predicoes == y_teste)
    erros   = np.sum(predicoes != y_teste)
    print(f"      Acertos: {acertos:,} ({acertos/len(y_teste)*100:.2f}%)")
    print(f"      Erros:   {erros:,} ({erros/len(y_teste)*100:.2f}%)")

    return probabilidades, predicoes


# =============================================================================
# PASSO 8: SALVAR O MODELO
# =============================================================================

def salvar_modelo(model):
    """
    Salva o modelo completo (arquitetura + pesos + configuração do otimizador).
    O formato .keras é o recomendado para TensorFlow 2.x.

    Para carregar depois:
        model = keras.models.load_model("mnist_mlp_model.keras")
        predicao = model.predict(nova_imagem)
    """
    print()
    print("[8/8] Salvando o modelo treinado...")

    model.save(MODELO_PATH)
    tamanho = os.path.getsize(MODELO_PATH) / 1024  # KB
    print(f"      Salvo em: {MODELO_PATH}")
    print(f"      Tamanho: {tamanho:.1f} KB")
    print()
    print("      Para usar depois:")
    print(f"      model = keras.models.load_model('{MODELO_PATH}')")
    print("      predicao = model.predict(nova_imagem)")


# =============================================================================
# GRÁFICOS: HISTÓRICO DE TREINAMENTO
# =============================================================================

def plotar_historico(history):
    """
    Exibe as curvas de loss e accuracy durante o treinamento.
    Ideal para diagnosticar overfitting ou underfitting.
    """
    epocas = range(1, len(history.history["loss"]) + 1)
    melhor = np.argmin(history.history["val_loss"]) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Loss ──
    ax1.plot(epocas, history.history["loss"],     "b-o", markersize=4,
             linewidth=2, label="Treino")
    ax1.plot(epocas, history.history["val_loss"], "r--o", markersize=4,
             linewidth=2, label="Validacao")
    ax1.axvline(x=melhor, color="green", linestyle=":", alpha=0.8,
                label=f"Melhor epoch ({melhor})")
    ax1.set_title("Loss por Epoca", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Accuracy ──
    ax2.plot(epocas, history.history["accuracy"],     "b-o", markersize=4,
             linewidth=2, label="Treino")
    ax2.plot(epocas, history.history["val_accuracy"], "r--o", markersize=4,
             linewidth=2, label="Validacao")
    ax2.axvline(x=melhor, color="green", linestyle=":", alpha=0.8,
                label=f"Melhor epoch ({melhor})")
    ax2.axhline(y=0.98, color="orange", linestyle="--", alpha=0.7,
                linewidth=1.5, label="Meta 98%")
    ax2.set_title("Acuracia por Epoca", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Acuracia")
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0.88, 1.01)

    plt.suptitle(
        "Historico de Treinamento — MLP no MNIST",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


# =============================================================================
# GRÁFICOS: PREDIÇÕES (ACERTOS E ERROS)
# =============================================================================

def plotar_predicoes(x_teste, y_teste, predicoes, probabilidades):
    """
    Grade de imagens com as predições da rede.
    Verde = acerto | Vermelho = erro
    """
    erros   = np.where(predicoes != y_teste)[0]
    acertos = np.where(predicoes == y_teste)[0]

    # Seleciona 8 acertos e 4 erros para exibir
    indices_acerto = acertos[:8]
    indices_erro   = erros[:4]
    todos = list(indices_acerto) + list(indices_erro)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()

    for i, idx in enumerate(todos):
        ax = axes[i]
        ax.imshow(x_teste[idx], cmap="gray")

        eh_acerto = predicoes[idx] == y_teste[idx]
        conf      = probabilidades[idx][predicoes[idx]] * 100
        cor       = "darkgreen" if eh_acerto else "darkred"
        status    = "CERTO" if eh_acerto else "ERRADO"

        ax.set_title(
            f"Real: {y_teste[idx]}  |  Pred: {predicoes[idx]}\n"
            f"{status} — Conf: {conf:.0f}%",
            fontsize=9, color=cor, fontweight="bold"
        )
        ax.axis("off")

        # Borda colorida
        for spine in ax.spines.values():
            spine.set_edgecolor("green" if eh_acerto else "red")
            spine.set_linewidth(3)

    plt.suptitle(
        "Predicoes do Modelo — Verde=Acerto  Vermelho=Erro",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


# =============================================================================
# GRÁFICOS: MATRIZ DE CONFUSÃO
# =============================================================================

def plotar_confusao(y_teste, predicoes):
    """
    Matriz de confusão normalizada — mostra onde a rede mais erra.
    Diagonal = acertos | Fora da diagonal = erros.
    """
    cm = confusion_matrix(y_teste, predicoes)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cm_norm,
        annot=True, fmt=".2f", cmap="Blues",
        xticklabels=range(10), yticklabels=range(10),
        ax=ax, linewidths=0.5,
        annot_kws={"size": 10}
    )

    ax.set_title(
        "Matriz de Confusao Normalizada\n"
        "(Proporcao de acertos e erros por digito real)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Predito pela rede", fontsize=11)
    ax.set_ylabel("Real (verdadeiro)", fontsize=11)

    plt.tight_layout()
    plt.show()

    # Relatório de classificação resumido
    print("\nResumo de desempenho por digito:")
    print(classification_report(
        y_teste, predicoes,
        target_names=[str(i) for i in range(10)]
    ))


# =============================================================================
# EXECUÇÃO PRINCIPAL — PIPELINE COMPLETO
# =============================================================================

if __name__ == "__main__":
    # ── Passo 1: Carregar os dados ──────────────────────────────────────────
    x_treino, y_treino, x_teste, y_teste = carregar_dados()

    # ── Passo 2: Preparar os dados ──────────────────────────────────────────
    x_treino_norm, x_teste_norm = preparar_dados(x_treino, x_teste)

    # ── Passo 3: Construir o modelo ─────────────────────────────────────────
    model = construir_modelo()

    # ── Passo 4: Compilar ───────────────────────────────────────────────────
    model = compilar_modelo(model)

    # ── Passo 5: Treinar ────────────────────────────────────────────────────
    history = treinar_modelo(model, x_treino_norm, y_treino)

    # ── Passo 6: Avaliar ────────────────────────────────────────────────────
    loss, accuracy = avaliar_modelo(model, x_teste_norm, y_teste)

    # ── Passo 7: Predições ──────────────────────────────────────────────────
    probabilidades, predicoes = obter_predicoes(model, x_teste_norm, y_teste)

    # ── Passo 8: Salvar o modelo ────────────────────────────────────────────
    salvar_modelo(model)

    # ── Gráficos em sequência ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print("GERANDO GRAFICOS FINAIS...")
    print("=" * 60)

    # Gráfico 1: Histórico de treinamento (loss + accuracy)
    plotar_historico(history)

    # Gráfico 2: Grade de predições (acertos e erros)
    plotar_predicoes(x_teste_norm, y_teste, predicoes, probabilidades)

    # Gráfico 3: Matriz de confusão
    plotar_confusao(y_teste, predicoes)

    # ── Conclusão final ─────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("TUTORIAL CONCLUIDO!")
    print("=" * 60)
    print()
    print(f"Resultado final: {accuracy*100:.2f}% de acuracia no conjunto de teste")
    print()
    print("O que voce aprendeu neste tutorial:")
    print("  1. O problema: reconhecer digitos escritos a mao (MNIST)")
    print("  2. Os dados: 70.000 imagens 28x28, pixels de 0 a 255")
    print("  3. Preparacao: normalizar e usar Flatten no modelo")
    print("  4. Arquitetura: Flatten + 2x(Dense+Dropout) + Softmax")
    print("  5. Treinamento: Adam + EarlyStopping + ReduceLROnPlateau")
    print("  6. Avaliacao: accuracy, matriz de confusao, F1-score")
    print("  7. Salvamento: model.save() para reutilizar depois")
    print()
    print("Proximos passos sugeridos:")
    print("  -> Testar com suas proprias imagens!")
    print("  -> Experimentar arquiteturas diferentes (mais/menos neuronios)")
    print("  -> Implementar uma CNN (Rede Neural Convolucional) — ainda melhor!")
    print("     (Convolucional considera a estrutura espacial da imagem)")
    print()
    print("Consulte os arquivos GO1055 a GO1060 para")
    print("explicacoes detalhadas de cada etapa.")
    print("=" * 60)
