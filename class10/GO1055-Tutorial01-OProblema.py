# =============================================================================
# Identificador: GO1055-Tutorial01-OProblema
# Aula 10 — MLP: Redes Neurais Multicamadas
# Tutorial Passo a Passo: Reconhecimento de Dígitos Escritos à Mão
# =============================================================================
#
# PARTE 1 DE 7 — O PROBLEMA
#
# Imagine que você trabalha nos Correios. Todo dia chegam milhares de envelopes
# com CEPs escritos à mão. Cada pessoa escreve de um jeito diferente:
# uns deixam o "1" parecendo "7", outros fazem o "0" parecido com "6".
#
# Como o computador pode aprender a ler esses dígitos automaticamente?
#
# A resposta: com uma Rede Neural treinada no dataset MNIST — uma coleção de
# 70.000 imagens de dígitos escritos por pessoas reais.
#
# Nesta parte você vai:
#   1. Carregar o dataset MNIST e entender sua estrutura
#   2. Visualizar exemplos reais de dígitos
#   3. Entender o que são "pixels" (os quadradinhos que formam a imagem)
#   4. Ver como os dígitos estão distribuídos no dataset
# =============================================================================

# Importamos o matplotlib ANTES de tudo — é a biblioteca para criar gráficos
import matplotlib
import matplotlib.pyplot as plt

# Este bloco detecta se estamos num Jupyter Notebook ou num terminal.
# No Jupyter, os gráficos aparecem inline. No terminal, abre uma janela.
try:
    get_ipython()
    matplotlib.use("module://matplotlib_inline.backend_inline")
except NameError:
    pass

# Numpy: biblioteca para trabalhar com arrays numéricos (matrizes, vetores)
import numpy as np

# TensorFlow/Keras: a biblioteca que usamos para construir redes neurais
# O Keras vem dentro do TensorFlow a partir da versão 2.x
import tensorflow as tf
from tensorflow import keras


# =============================================================================
# CARREGANDO O DATASET MNIST
# =============================================================================

def carregar_mnist():
    """
    O Keras já vem com o MNIST embutido — não precisamos baixar nada!
    Ele já divide automaticamente em treino (60.000) e teste (10.000).

    Por que essa divisão?
    - Treino: a rede aprende com esses dados
    - Teste: verificamos se a rede aprendeu de verdade (dados que ela nunca viu)

    Se usássemos os mesmos dados para treinar e testar, seria como dar as
    respostas da prova para o aluno estudar — ele seria aprovado, mas não
    teria aprendido nada de novo.
    """
    print("=" * 60)
    print("TUTORIAL AULA 10 — Reconhecimento de Digitos MNIST")
    print("=" * 60)
    print()
    print("Carregando o dataset MNIST...")
    print("(Na primeira vez, faz download automatico ~11MB)")
    print()

    # keras.datasets.mnist.load_data() retorna dois pares (imagens, labels)
    (x_treino, y_treino), (x_teste, y_teste) = keras.datasets.mnist.load_data()

    return x_treino, y_treino, x_teste, y_teste


def explorar_dataset(x_treino, y_treino, x_teste, y_teste):
    """
    Vamos entender o que exatamente carregamos.

    x_treino: array com as IMAGENS de treino (os pixels)
    y_treino: array com os LABELS de treino (os rótulos — qual dígito é)
    x_teste:  array com as IMAGENS de teste
    y_teste:  array com os LABELS de teste
    """
    print("ESTRUTURA DO DATASET:")
    print("-" * 40)

    # .shape retorna as dimensões do array
    # (60000, 28, 28) significa: 60000 imagens, cada uma com 28 linhas x 28 colunas
    print(f"Imagens de treino: {x_treino.shape}")
    print(f"  -> {x_treino.shape[0]} imagens")
    print(f"  -> Cada imagem tem {x_treino.shape[1]}x{x_treino.shape[2]} pixels")

    print(f"\nLabels de treino: {y_treino.shape}")
    print(f"  -> Um numero (0-9) para cada imagem")

    print(f"\nImagens de teste:  {x_teste.shape}")
    print(f"Labels de teste:   {y_teste.shape}")

    print(f"\nTotal de imagens:  {x_treino.shape[0] + x_teste.shape[0]:,}")

    # Tipo dos dados: uint8 = inteiro sem sinal de 8 bits
    # Valores de 0 a 255 (porque 2^8 = 256 valores possíveis)
    print(f"\nTipo de dado:      {x_treino.dtype}")
    print(f"Valor minimo:      {x_treino.min()}  (pixel completamente preto)")
    print(f"Valor maximo:      {x_treino.max()}  (pixel completamente branco)")

    print(f"\nDigitos possiveis: {sorted(np.unique(y_treino))} (classes 0 a 9)")


def visualizar_exemplos(x_treino, y_treino):
    """
    Vamos ver como esses dígitos se parecem!

    Cada imagem é uma grade de 28x28 = 784 quadradinhos (pixels).
    Cada quadradinho tem um valor de 0 (preto) a 255 (branco).

    Quando você vê uma imagem no computador, está olhando para uma
    tabela de números — a rede neural aprende exatamente isso!
    """
    print("\n" + "=" * 60)
    print("VISUALIZANDO EXEMPLOS DO DATASET")
    print("=" * 60)

    # Escolhemos um exemplo de cada dígito (0 a 9) para visualizar
    # np.where retorna os índices onde a condição é verdadeira
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle(
        "Exemplos do Dataset MNIST — Um de Cada Digito",
        fontsize=16, fontweight="bold", y=1.02
    )

    for digito in range(10):
        # Encontra o primeiro índice onde o label é igual ao dígito atual
        indice = np.where(y_treino == digito)[0][0]

        # Calcula posição na grade 2x5
        linha = digito // 5
        coluna = digito % 5
        ax = axes[linha][coluna]

        # Mostra a imagem em escala de cinza (cmap='gray')
        # vmin=0, vmax=255 garante que a escala de cores seja consistente
        ax.imshow(x_treino[indice], cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Digito: {digito}", fontsize=13, fontweight="bold")
        ax.axis("off")  # Remove os eixos para ficar mais limpo

    plt.tight_layout()
    plt.show()

    print("-> Voce pode ver como cada pessoa escreve diferente!")
    print("   A rede neural vai aprender os padroes comuns de cada digito.")


def mostrar_pixels(x_treino, y_treino):
    """
    Vamos ver o que está "por dentro" de uma imagem.

    Uma imagem digital é, na prática, uma tabela de números.
    Para o computador, a imagem do dígito "5" é uma matriz 28x28
    onde cada célula guarda um número entre 0 e 255.

    0   = pixel preto (sem tinta)
    128 = pixel cinza (meia tinta)
    255 = pixel branco (tinta máxima)

    A rede neural vai trabalhar diretamente com esses números!
    """
    print("\n" + "=" * 60)
    print("OS PIXELS POR DENTRO DE UMA IMAGEM")
    print("=" * 60)

    # Pegamos a primeira imagem do dataset para examinar
    imagem = x_treino[0]
    label = y_treino[0]

    print(f"Imagem selecionada: digito '{label}'")
    print(f"Tamanho da imagem: {imagem.shape[0]}x{imagem.shape[1]} = {imagem.size} pixels")
    print()

    # Mostramos apenas o centro da imagem (8x8) onde estão os pixels mais escuros
    # A maioria das bordas é preta (zero), então o centro é mais interessante
    print("Submatriz 8x8 (centro da imagem — pixels 10 a 18):")
    print("Valores de 0 (preto) a 255 (branco):")
    print()

    # Formatamos cada número com 3 dígitos para alinhar colunas
    submatriz = imagem[10:18, 10:18]
    for linha in submatriz:
        print("  " + "  ".join(f"{pixel:3d}" for pixel in linha))

    print()
    print("Repare: onde tem tinta (o traço do digito), o numero e alto (perto de 255)")
    print("Onde e fundo branco, o numero e baixo (perto de 0)")
    print("(O MNIST usa fundo preto e tracos brancos — ao contrario do papel normal)")


def mostrar_distribuicao(y_treino):
    """
    Importante: quantas imagens de cada dígito temos?

    Se tivermos muito mais "1" do que "9", a rede pode aprender
    mal para dígitos raros. Um dataset "balanceado" tem quantidades
    parecidas de cada classe.
    """
    print("\n" + "=" * 60)
    print("DISTRIBUICAO DOS DIGITOS NO DATASET")
    print("=" * 60)

    # np.unique com return_counts=True retorna os valores únicos e suas contagens
    digitos, contagens = np.unique(y_treino, return_counts=True)

    print("Quantidade de exemplos por digito (treino):")
    for d, c in zip(digitos, contagens):
        barra = "#" * (c // 100)  # Cada # representa 100 exemplos
        print(f"  Digito {d}: {c:,} exemplos  {barra}")

    # Visualização como histograma
    fig, ax = plt.subplots(figsize=(12, 5))

    cores = plt.cm.tab10(np.linspace(0, 1, 10))  # 10 cores diferentes
    barras = ax.bar(digitos, contagens, color=cores, edgecolor="white", linewidth=0.8)

    # Adiciona o número exato em cima de cada barra
    for barra, contagem in zip(barras, contagens):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 30,
            f"{contagem:,}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.set_title(
        "Distribuicao dos Digitos no Dataset MNIST (Treino)",
        fontsize=14, fontweight="bold"
    )
    ax.set_xlabel("Digito", fontsize=12)
    ax.set_ylabel("Numero de Exemplos", fontsize=12)
    ax.set_xticks(range(10))
    ax.set_ylim(0, max(contagens) * 1.15)

    # Linha de referência (média)
    media = np.mean(contagens)
    ax.axhline(y=media, color="red", linestyle="--", alpha=0.7, linewidth=1.5)
    ax.text(9.5, media + 50, f"Media: {media:.0f}", color="red", ha="right", fontsize=10)

    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\nMedia por digito: {np.mean(contagens):.0f} imagens")
    print(f"Desvio padrao:    {np.std(contagens):.0f} imagens")
    print("-> Dataset bem balanceado! Cada digito tem quantidade similar de exemplos.")


def resumo_parte1():
    """
    Recapitula o que aprendemos e prepara para o próximo passo.
    """
    print()
    print("=" * 60)
    print("RESUMO DA PARTE 1 — O QUE APRENDEMOS:")
    print("=" * 60)
    print()
    print("1. O dataset MNIST tem 70.000 imagens de digitos (0-9)")
    print("   -> 60.000 para treino, 10.000 para teste")
    print()
    print("2. Cada imagem e uma grade de 28x28 = 784 pixels")
    print("   -> Cada pixel e um numero de 0 (preto) a 255 (branco)")
    print()
    print("3. O dataset e bem balanceado (~6.000 exemplos por digito)")
    print("   -> Bom para treinar uma rede neural justa")
    print()
    print("4. A rede neural vai aprender a mapear:")
    print("   784 numeros (pixels) -> 1 numero (qual digito e)")
    print()
    print("-" * 60)
    print("PROXIMA PARTE: Entendendo os Dados em Profundidade")
    print("   -> Por que normalizar? Como visualizar os padroes?")
    print("   -> Execute: GO1056-Tutorial02-EntendendoOsDados.py")
    print("=" * 60)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Carrega os dados
    x_treino, y_treino, x_teste, y_teste = carregar_mnist()

    # Explora a estrutura do dataset
    explorar_dataset(x_treino, y_treino, x_teste, y_teste)

    # Visualiza exemplos (gráfico 1: grade de imagens)
    visualizar_exemplos(x_treino, y_treino)

    # Mostra o que são pixels
    mostrar_pixels(x_treino, y_treino)

    # Mostra distribuição dos dígitos (gráfico 2: histograma)
    mostrar_distribuicao(y_treino)

    # Recapitula o que aprendemos
    resumo_parte1()
