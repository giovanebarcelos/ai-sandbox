# =============================================================================
# Identificador: GO1057-Tutorial03-PreparandoOsDados
# Aula 10 — MLP: Redes Neurais Multicamadas
# Tutorial Passo a Passo: Reconhecimento de Dígitos Escritos à Mão
# =============================================================================
#
# PARTE 3 DE 7 — PREPARANDO OS DADOS
#
# Os dados brutos do MNIST precisam ser transformados antes de entrar
# na rede neural. Esta etapa é chamada de "pré-processamento" e é
# FUNDAMENTAL para um bom desempenho.
#
# As 4 etapas desta parte:
#   1. Normalização:    0-255  →  0.0-1.0
#   2. Reshape:         28x28  →  784 (vetor linear)
#   3. One-hot encoding: "3"   →  [0,0,0,1,0,0,0,0,0,0]
#   4. Divisão treino/validação (já vem separado no MNIST)
#
# ANALOGIA GERAL: É como preparar ingredientes antes de cozinhar.
# Você não coloca o frango no forno com a embalagem — precisa limpar,
# temperar, cortar... Os dados precisam do mesmo cuidado!
# =============================================================================

# matplotlib SEMPRE no nível do módulo
import matplotlib
import matplotlib.pyplot as plt

# Configura backend conforme o ambiente de execução
try:
    get_ipython()
    matplotlib.use("module://matplotlib_inline.backend_inline")
except NameError:
    pass

import numpy as np
import tensorflow as tf
from tensorflow import keras


# =============================================================================
# CARREGANDO OS DADOS
# =============================================================================

def carregar_dados():
    """Carrega o MNIST para trabalharmos com os dados brutos nesta aula."""
    print("=" * 60)
    print("TUTORIAL AULA 10 — PARTE 3: Preparando os Dados")
    print("=" * 60)
    print()
    (x_treino, y_treino), (x_teste, y_teste) = keras.datasets.mnist.load_data()
    print(f"Dados carregados com sucesso!")
    print(f"  x_treino shape: {x_treino.shape} | dtype: {x_treino.dtype}")
    print(f"  y_treino shape: {y_treino.shape} | dtype: {y_treino.dtype}")
    return x_treino, y_treino, x_teste, y_teste


# =============================================================================
# ETAPA 1: NORMALIZAÇÃO
# =============================================================================

def etapa1_normalizacao(x_treino, x_teste):
    """
    ETAPA 1: Normalização dos pixels (0-255 → 0.0-1.0)

    POR QUÊ É NECESSÁRIO?
    ─────────────────────
    ANALOGIA: Pense em uma corrida com dois corredores.
    Um corre 100 metros, o outro corre 1 quilômetro. Se você
    quiser comparar a velocidade dos dois, precisa converter
    tudo para a mesma unidade (km/h, por exemplo).

    O mesmo vale para as entradas da rede neural:
    - Sem normalização: pixel 200 "pesa" 200x mais que pixel 1
    - Com normalização: pixel 0.78 e pixel 0.004 estão na mesma escala

    Isso torna o aprendizado MUITO mais estável.

    COMO FAZER:
    ─────────────
    Dividimos por 255 porque:
    - O maior valor possível é 255
    - 255 / 255 = 1.0 (máximo vira 1.0)
    - 0 / 255 = 0.0 (mínimo vira 0.0)
    """
    print("\n" + "=" * 60)
    print("ETAPA 1: NORMALIZACAO (0-255 -> 0.0-1.0)")
    print("=" * 60)

    print("\nANTES da normalizacao:")
    print(f"  Tipo:  {x_treino.dtype}")
    print(f"  Min:   {x_treino.min()}")
    print(f"  Max:   {x_treino.max()}")
    print(f"  Media: {x_treino.mean():.2f}")

    # astype("float32") converte de inteiro para float de 32 bits
    # float32 usa menos memória que float64 e é suficiente para redes neurais
    x_treino_norm = x_treino.astype("float32") / 255.0
    x_teste_norm  = x_teste.astype("float32")  / 255.0

    print("\nDEPOIS da normalizacao:")
    print(f"  Tipo:  {x_treino_norm.dtype}")
    print(f"  Min:   {x_treino_norm.min():.4f}")
    print(f"  Max:   {x_treino_norm.max():.4f}")
    print(f"  Media: {x_treino_norm.mean():.4f}")

    print("\nO que mudou:")
    print("  Antes: pixel = 128  -->  Depois: pixel = 0.502")
    print("  Antes: pixel = 255  -->  Depois: pixel = 1.000")
    print("  Antes: pixel =   0  -->  Depois: pixel = 0.000")

    return x_treino_norm, x_teste_norm


# =============================================================================
# ETAPA 2: RESHAPE (28x28 → 784)
# =============================================================================

def etapa2_reshape(x_treino_norm, x_teste_norm):
    """
    ETAPA 2: Reshape — Transformar a grade 2D em um vetor 1D

    POR QUÊ É NECESSÁRIO?
    ─────────────────────
    A camada Dense (a que vamos usar) espera um VETOR (lista linear)
    como entrada, não uma matriz 2D.

    ANALOGIA: Pense em uma foto tirada do alto de um campo de futebol.
    Você vê um retângulo com jogadores. Agora "desenrole" esse campo
    como se fosse uma fita de papel — todos os jogadores em uma linha só.

    É exatamente isso que fazemos: pegamos a grade 28x28 e
    "desenrolamos" em uma lista de 784 números.

    28 linhas × 28 colunas = 784 pixels no total

    DETALHE IMPORTANTE:
    ───────────────────
    No Keras, podemos usar a camada Flatten() dentro do modelo
    para fazer isso automaticamente. Mas é importante entender
    o que está acontecendo por baixo.
    """
    print("\n" + "=" * 60)
    print("ETAPA 2: RESHAPE (28x28 -> vetor de 784)")
    print("=" * 60)

    print(f"\nFORMA ANTES do reshape: {x_treino_norm.shape}")
    print("  Interpretacao: 60000 imagens, cada uma 28x28 pixels")
    print("  (imagem, linha, coluna)")

    # reshape(-1, 784):
    #   -1 = "calcule automaticamente" (preserva 60000)
    #   784 = 28*28 = número de pixels por imagem
    x_treino_flat = x_treino_norm.reshape(-1, 784)
    x_teste_flat  = x_teste_norm.reshape(-1, 784)

    print(f"\nFORMA DEPOIS do reshape: {x_treino_flat.shape}")
    print("  Interpretacao: 60000 imagens, cada uma com 784 numeros em linha")
    print("  (imagem, pixel)")

    print("\nComo funciona o reshape:")
    print("  [pixel(0,0), pixel(0,1), ..., pixel(0,27),  <- linha 0")
    print("   pixel(1,0), pixel(1,1), ..., pixel(1,27),  <- linha 1")
    print("   ...")
    print("   pixel(27,0), ..., pixel(27,27)]             <- linha 27")
    print("                                                  = 784 valores")

    return x_treino_flat, x_teste_flat


# =============================================================================
# ETAPA 3: ONE-HOT ENCODING
# =============================================================================

def etapa3_one_hot(y_treino, y_teste):
    """
    ETAPA 3: One-Hot Encoding dos Labels

    POR QUÊ É NECESSÁRIO?
    ─────────────────────
    Os labels originais são números: 0, 1, 2, ..., 9
    Poderíamos usar esses números diretamente, mas isso cria um
    problema: a rede poderia interpretar que "8 > 5" tem algum
    significado, como se o dígito 8 fosse "maior" ou "mais importante"
    que o dígito 5. Mas isso não faz sentido! São categorias iguais.

    ANALOGIA — Formulário de múltipla escolha:
    Imagine que a rede precisa "marcar" qual dígito é.
    Em vez de escrever o número "3", ela marca um círculo:

    [_] 0  [_] 1  [_] 2  [X] 3  [_] 4  [_] 5  [_] 6  [_] 7  [_] 8  [_] 9

    Em formato de array: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

    Isso é o one-hot encoding: apenas UMA posição é 1 (hot),
    todas as outras são 0 (cold).

    NOTA: O Keras tem loss='sparse_categorical_crossentropy' que
    aceita labels inteiros diretamente (sem one-hot). Mas é importante
    entender o conceito para quando você precisar usar outros frameworks.
    """
    print("\n" + "=" * 60)
    print("ETAPA 3: ONE-HOT ENCODING DOS LABELS")
    print("=" * 60)

    print("\nANTES do one-hot encoding:")
    print(f"  Shape: {y_treino.shape}")
    print(f"  Tipo:  {y_treino.dtype}")
    print(f"  Primeiros 10 labels: {y_treino[:10]}")

    # keras.utils.to_categorical converte inteiros para one-hot
    # num_classes=10 especifica que temos 10 classes (dígitos 0-9)
    y_treino_oh = keras.utils.to_categorical(y_treino, num_classes=10)
    y_teste_oh  = keras.utils.to_categorical(y_teste,  num_classes=10)

    print("\nDEPOIS do one-hot encoding:")
    print(f"  Shape: {y_treino_oh.shape}  <- agora temos 10 colunas!")
    print(f"  Tipo:  {y_treino_oh.dtype}")

    print("\nExemplos de conversao:")
    for i in range(5):
        label_original = y_treino[i]
        label_oh = y_treino_oh[i]
        oh_str = "[" + ", ".join(str(int(v)) for v in label_oh) + "]"
        print(f"  Digito {label_original}  ->  {oh_str}")

    print("\nCada posicao do array corresponde a um digito:")
    print("  Posicao: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]")
    print("  Digito 3: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]")
    print("  Digito 7: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]")

    return y_treino_oh, y_teste_oh


# =============================================================================
# ETAPA 4: DIVISÃO TREINO / VALIDAÇÃO / TESTE
# =============================================================================

def etapa4_divisao(x_treino_flat, y_treino_oh, x_teste_flat, y_teste_oh):
    """
    ETAPA 4: Entendendo a Divisão dos Dados

    O MNIST já vem pré-dividido em treino e teste. Mas por que essa divisão?

    TREINO   (60.000): A rede aprende com esses dados
    VALIDAÇÃO (parte do treino): Monitora se está generalizando bem
    TESTE    (10.000): Avaliação FINAL — dados que a rede nunca viu

    ANALOGIA:
    - Treino = o caderno de exercícios do aluno
    - Validação = os exercícios extras para praticar
    - Teste = a prova final (o aluno não vê antes)

    Se o aluno tivesse as respostas da prova final enquanto estuda,
    seria trapaça! O mesmo vale para a rede neural: se ela "visse"
    os dados de teste durante o treino, os resultados seriam ilusórios.

    VALIDAÇÃO DENTRO DO TREINO:
    Durante o fit(), usamos validation_split=0.1 para reservar
    automaticamente 10% do treino para validação. O Keras monitora
    esse subconjunto a cada época para detectar overfitting.
    """
    print("\n" + "=" * 60)
    print("ETAPA 4: DIVISAO TREINO / VALIDACAO / TESTE")
    print("=" * 60)

    print("\nDivisao do MNIST:")
    print(f"  Treino:     {x_treino_flat.shape[0]:,} imagens")
    print(f"  Teste:      {x_teste_flat.shape[0]:,} imagens")

    print("\nDurante o treinamento (validation_split=0.1):")
    n_val = int(x_treino_flat.shape[0] * 0.1)
    n_fit = x_treino_flat.shape[0] - n_val
    print(f"  Treino efetivo: {n_fit:,} imagens (90%)")
    print(f"  Validacao:      {n_val:,} imagens (10%)")

    print("\nPapel de cada conjunto:")
    print("  TREINO:    A rede APRENDE com esses dados (ajusta os pesos)")
    print("  VALIDACAO: A rede NAO aprende, apenas checamos o desempenho")
    print("             -> Detecta overfitting (quando a rede 'decora' os dados)")
    print("  TESTE:     Avaliacao FINAL — resultados para publicar/reportar")

    return x_treino_flat, y_treino_oh, x_teste_flat, y_teste_oh


# =============================================================================
# VISUALIZAÇÃO: ANTES vs DEPOIS DE CADA ETAPA
# =============================================================================

def visualizar_pipeline(x_treino, y_treino,
                        x_treino_flat, y_treino_oh):
    """
    Visualização comparativa de cada etapa de preparação.
    Um bom gráfico vale mais do que mil linhas de código!
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        "Pipeline de Preparacao dos Dados — Antes e Depois de Cada Etapa",
        fontsize=14, fontweight="bold"
    )

    # ── Etapa 1: Imagem bruta (antes) ──
    ax = axes[0]
    ax.imshow(x_treino[0], cmap="gray", vmin=0, vmax=255)
    ax.set_title(
        f"ENTRADA BRUTA\n28x28 pixels\nValores: 0 a 255\ndtype: {x_treino.dtype}",
        fontsize=10
    )
    ax.axis("off")

    # ── Etapa 2: Imagem normalizada ──
    ax = axes[1]
    x_norm = x_treino[0].astype("float32") / 255.0
    ax.imshow(x_norm, cmap="gray", vmin=0, vmax=1)
    ax.set_title(
        f"ETAPA 1: NORMALIZADA\n28x28 pixels\nValores: 0.0 a 1.0\ndtype: float32",
        fontsize=10
    )
    ax.axis("off")

    # ── Etapa 3: Vetor linearizado ──
    ax = axes[2]
    vetor = x_treino_flat[0]  # já é 784
    # Exibe como heatmap horizontal
    ax.imshow(vetor.reshape(1, -1), cmap="gray", aspect="auto")
    ax.set_title(
        f"ETAPA 2: FLATTEN\nVetor de 784 numeros\n(28x28 linearizado)\nShape: (784,)",
        fontsize=10
    )
    ax.set_xlabel("Posicao do pixel (0-783)", fontsize=8)
    ax.set_yticks([])

    # ── Etapa 4: One-hot label ──
    ax = axes[3]
    label = y_treino[0]
    oh = y_treino_oh[0]  # array de 10 valores
    cores = ["#1565c0" if v == 1 else "#e3f2fd" for v in oh]
    barras = ax.bar(range(10), oh, color=cores, edgecolor="white", linewidth=0.8)
    ax.set_title(
        f"ETAPA 3: ONE-HOT\nDigito '{label}' -> posicao {label} = 1\nRestante = 0",
        fontsize=10
    )
    ax.set_xlabel("Digito (0-9)", fontsize=9)
    ax.set_ylabel("Valor (0 ou 1)", fontsize=9)
    ax.set_xticks(range(10))
    ax.set_ylim(0, 1.3)
    for i, (b, v) in enumerate(zip(barras, oh)):
        if v > 0:
            ax.text(b.get_x() + 0.4, 1.05, "1", ha="center", fontsize=11, fontweight="bold", color="#1565c0")

    plt.tight_layout()
    plt.show()


# =============================================================================
# FUNÇÃO PRINCIPAL — RETORNA DADOS PREPARADOS
# =============================================================================

def preparar_dados_completo():
    """
    Executa todo o pipeline de preparação e retorna os dados prontos.
    Esta função é a que outros scripts podem importar e usar.
    """
    (x_treino, y_treino), (x_teste, y_teste) = keras.datasets.mnist.load_data()

    # Etapa 1: Normalização
    x_treino_norm = x_treino.astype("float32") / 255.0
    x_teste_norm  = x_teste.astype("float32")  / 255.0

    # Etapa 2: Reshape (a camada Flatten do Keras faz isso automaticamente,
    # mas fazemos manualmente aqui para fins didáticos)
    x_treino_flat = x_treino_norm.reshape(-1, 784)
    x_teste_flat  = x_teste_norm.reshape(-1, 784)

    # Labels inteiros (para sparse_categorical_crossentropy)
    # Mantemos y_treino e y_teste como estão (0-9)

    return (x_treino_flat, y_treino), (x_teste_flat, y_teste)


def resumo_parte3():
    """Recapitula as etapas de preparação."""
    print()
    print("=" * 60)
    print("RESUMO DA PARTE 3 — PIPELINE DE PREPARACAO:")
    print("=" * 60)
    print()
    print("ETAPA 1 — NORMALIZACAO")
    print("  x / 255.0  ->  valores de 0.0 a 1.0")
    print("  Por que: estabiliza o treinamento da rede neural")
    print()
    print("ETAPA 2 — RESHAPE")
    print("  28x28  ->  vetor de 784 numeros")
    print("  Por que: a camada Dense espera um vetor 1D")
    print("  (o Keras pode fazer isso com Flatten() automaticamente)")
    print()
    print("ETAPA 3 — ONE-HOT ENCODING")
    print("  '3'  ->  [0,0,0,1,0,0,0,0,0,0]")
    print("  Por que: evita que a rede interprete ordem entre classes")
    print()
    print("ETAPA 4 — DIVISAO")
    print("  60k treino (+ 10% validacao interna) | 10k teste")
    print("  Por que: avaliar se a rede GENERALIZA, nao apenas memoriza")
    print()
    print("-" * 60)
    print("PROXIMA PARTE: Construindo a Rede Neural")
    print("   -> Dense, ReLU, Dropout, Softmax explicados")
    print("   -> Execute: GO1058-Tutorial04-ConstruindoARedeNeural.py")
    print("=" * 60)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Carrega os dados brutos
    x_treino, y_treino, x_teste, y_teste = carregar_dados()

    # Etapa 1: Normalização
    x_treino_norm, x_teste_norm = etapa1_normalizacao(x_treino, x_teste)

    # Etapa 2: Reshape
    x_treino_flat, x_teste_flat = etapa2_reshape(x_treino_norm, x_teste_norm)

    # Etapa 3: One-hot encoding
    y_treino_oh, y_teste_oh = etapa3_one_hot(y_treino, y_teste)

    # Etapa 4: Divisão
    etapa4_divisao(x_treino_flat, y_treino_oh, x_teste_flat, y_teste_oh)

    # Gráfico: pipeline visual
    visualizar_pipeline(x_treino, y_treino, x_treino_flat, y_treino_oh)

    # Resumo
    resumo_parte3()
