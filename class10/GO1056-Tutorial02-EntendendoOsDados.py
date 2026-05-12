# =============================================================================
# Identificador: GO1056-Tutorial02-EntendendoOsDados
# Aula 10 — MLP: Redes Neurais Multicamadas
# Tutorial Passo a Passo: Reconhecimento de Dígitos Escritos à Mão
# =============================================================================
#
# PARTE 2 DE 7 — ENTENDENDO OS DADOS
#
# Antes de treinar qualquer rede neural, precisamos ENTENDER os dados.
# Perguntas que devemos responder:
#   - O que cada número no pixel significa?
#   - Por que precisamos normalizar os valores?
#   - O que a rede vai aprender de "especial" em cada dígito?
#   - Dígitos iguais realmente se parecem entre si?
#
# Esta parte foca na análise visual dos dados — uma etapa que muitos
# tutoriais pulam, mas que é essencial para entender o que a rede aprende.
# =============================================================================

# matplotlib e plt SEMPRE importados no nível do módulo
import matplotlib
import matplotlib.pyplot as plt

# Detecta ambiente (Jupyter vs terminal) e configura o backend apropriado
try:
    get_ipython()
    matplotlib.use("module://matplotlib_inline.backend_inline")
except NameError:
    pass

import numpy as np
import tensorflow as tf
from tensorflow import keras


# =============================================================================
# CARREGANDO OS DADOS (mesmo processo da Parte 1)
# =============================================================================

def carregar_dados():
    """
    Carregamos o MNIST novamente. Em um projeto real, você faria isso
    uma vez e reutilizaria. Aqui repetimos para que cada tutorial seja
    completo e possa ser executado de forma independente.
    """
    print("=" * 60)
    print("TUTORIAL AULA 10 — PARTE 2: Entendendo os Dados")
    print("=" * 60)
    print()
    (x_treino, y_treino), (x_teste, y_teste) = keras.datasets.mnist.load_data()
    print(f"Dados carregados: {x_treino.shape[0]:,} treino, {x_teste.shape[0]:,} teste")
    return x_treino, y_treino, x_teste, y_teste


# =============================================================================
# O QUE CADA PIXEL SIGNIFICA
# =============================================================================

def explicar_pixels(x_treino, y_treino):
    """
    Um pixel é a menor unidade de uma imagem digital.
    No MNIST, cada pixel armazena a intensidade de cinza:
        0   = preto absoluto (sem tinta, fundo)
        128 = cinza médio (meio tom)
        255 = branco absoluto (tinta máxima, o traço)

    Por que valores até 255? Porque usamos 8 bits por pixel:
        2^8 = 256 valores possíveis (0 a 255)
    Isso economiza memória e é o padrão para imagens em tons de cinza.
    """
    print("\n" + "=" * 60)
    print("O QUE CADA NUMERO NO PIXEL SIGNIFICA")
    print("=" * 60)

    imagem = x_treino[0]
    label = y_treino[0]

    print(f"\nImagem do digito '{label}':")
    print(f"  Pixels pretos (valor 0):   {np.sum(imagem == 0):,} pixels ({np.mean(imagem == 0)*100:.1f}% da imagem)")
    print(f"  Pixels cinzas (1-254):     {np.sum((imagem > 0) & (imagem < 255)):,} pixels")
    print(f"  Pixels brancos (255):      {np.sum(imagem == 255):,} pixels")
    print()
    print("  Fundo = 0 (preto) — onde nao tem tinta")
    print("  Traco = ~255 (branco/cinza claro) — onde a caneta passou")
    print()
    print("  Nota: MNIST usa fundo PRETO e tracos CLAROS")
    print("  (ao contrario do papel — fundo branco com tinta escura)")


# =============================================================================
# POR QUE NORMALIZAR? A ANALOGIA DAS NOTAS
# =============================================================================

def explicar_normalizacao(x_treino, y_treino):
    """
    PROBLEMA: Os pixels têm valores de 0 a 255.
    Isso parece inofensivo, mas causa um problema sério para a rede neural.

    ANALOGIA — Imagine que você está comparando alunos usando:
        - Notas de Matemática: de 0 a 10
        - Renda familiar: de 0 a 20.000

    Se você somar essas variáveis para uma análise, a renda vai
    DOMINAR completamente o resultado — os valores maiores têm
    mais peso, mesmo que não sejam mais importantes!

    O mesmo acontece com os pixels: valores de 0 a 255 são "grandes"
    para a rede neural. Os pesos ficam instáveis e o treinamento
    fica lento ou não converge.

    SOLUÇÃO: Normalizar para 0.0 a 1.0 dividindo por 255.
    Agora todos os pixels têm a mesma "escala" de importância.
    """
    print("\n" + "=" * 60)
    print("POR QUE NORMALIZAR OS PIXELS?")
    print("=" * 60)

    print()
    print("ANTES da normalizacao:")
    print(f"  Valor minimo: {x_treino.min()}")
    print(f"  Valor maximo: {x_treino.max()}")
    print(f"  Media global: {x_treino.mean():.2f}")
    print(f"  Tipo de dado: {x_treino.dtype}")

    # Normalizamos dividindo por 255.0
    # O .0 é importante para converter de inteiro para float!
    x_normalizado = x_treino.astype("float32") / 255.0

    print()
    print("DEPOIS da normalizacao (dividido por 255):")
    print(f"  Valor minimo: {x_normalizado.min():.4f}")
    print(f"  Valor maximo: {x_normalizado.max():.4f}")
    print(f"  Media global: {x_normalizado.mean():.4f}")
    print(f"  Tipo de dado: {x_normalizado.dtype}")

    print()
    print("Por que isso ajuda a rede neural?")
    print("  1. Valores pequenos (0-1) evitam explosao de gradientes")
    print("     (os numeros nao ficam enormes durante os calculos)")
    print("  2. O treinamento converge mais rapido e de forma mais estavel")
    print("  3. A funcao de ativacao ReLU funciona melhor com valores pequenos")

    return x_normalizado


# =============================================================================
# HEATMAP DE PIXELS COM VALORES NUMERADOS
# =============================================================================

def visualizar_heatmap(x_treino, y_treino):
    """
    Vamos visualizar os pixels como um "mapa de calor" numerado.
    Isso mostra exatamente o que a rede neural "vê" quando olha para
    uma imagem — uma tabela de números, não uma figura como nós vemos!

    Usaremos apenas 14x14 (metade da imagem original) para os números
    caberem na tela sem ficar muito pequenos.
    """
    print("\n" + "=" * 60)
    print("HEATMAP — COMO A REDE NEURAL VE UMA IMAGEM")
    print("=" * 60)

    # Pega um exemplo do dígito "3" — tem traços interessantes
    idx_3 = np.where(y_treino == 3)[0][0]
    imagem_3 = x_treino[idx_3]

    # Cortamos a região central (14x14) onde o dígito aparece
    # Área 7:21, 7:21 captura a parte central da imagem 28x28
    subimagem = imagem_3[7:21, 7:21]
    sub_normalizada = subimagem / 255.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Gráfico 1: Imagem original (esquerda) ──
    axes[0].imshow(imagem_3, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title(f'Imagem original (28x28)\nDigito "3"', fontsize=13, fontweight="bold")
    axes[0].axis("off")

    # Marca a área que vamos ampliar
    import matplotlib.patches as patches
    rect = patches.Rectangle(
        (6.5, 6.5), 14, 14,
        linewidth=2, edgecolor="red", facecolor="none"
    )
    axes[0].add_patch(rect)
    axes[0].text(7, 5.5, "Regiao ampliada ->", color="red", fontsize=9, fontweight="bold")

    # ── Gráfico 2: Heatmap numerado (direita) ──
    im = axes[1].imshow(sub_normalizada, cmap="YlOrRd", vmin=0, vmax=1)

    # Adiciona o valor de cada pixel na célula
    for i in range(14):
        for j in range(14):
            valor = sub_normalizada[i, j]
            # Texto branco para pixels escuros, preto para pixels claros
            cor_texto = "white" if valor > 0.5 else "black"
            axes[1].text(
                j, i, f"{valor:.1f}",
                ha="center", va="center",
                fontsize=7, color=cor_texto, fontweight="bold"
            )

    plt.colorbar(im, ax=axes[1], shrink=0.8)
    axes[1].set_title(
        'Heatmap 14x14 (regiao central)\nValores normalizados 0.0-1.0',
        fontsize=13, fontweight="bold"
    )
    axes[1].set_xlabel("Coluna (pixel)", fontsize=10)
    axes[1].set_ylabel("Linha (pixel)", fontsize=10)

    plt.suptitle(
        'Como a Rede Neural "Ve" uma Imagem — Uma Tabela de Numeros',
        fontsize=15, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.show()

    print("-> A rede neural nao ve um '3' bonito como nos vemos.")
    print("   Ela ve uma tabela de 784 numeros (0.0 a 1.0).")
    print("   O aprendizado consiste em descobrir QUAIS combinacoes")
    print("   desses numeros indicam cada digito!")


# =============================================================================
# OVERLAY — DÍGITOS IGUAIS SE PARECEM?
# =============================================================================

def visualizar_overlay(x_treino, y_treino):
    """
    Pergunta importante: se pegarmos 10 imagens diferentes do dígito "5",
    os pixels se parecerão entre si?

    SE SIM: a rede neural conseguirá aprender um "padrão" do "5"
    SE NÃO: vai ser muito difícil aprender qualquer coisa!

    Vamos visualizar a sobreposição (overlay) de múltiplos exemplos
    do mesmo dígito. A média dos pixels vai revelar onde o traço
    costuma aparecer — esse é o "padrão" que a rede vai aprender!
    """
    print("\n" + "=" * 60)
    print("DIGITOS IGUAIS TEM PIXELS PARECIDOS?")
    print("=" * 60)

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle(
        "Media de 200 Exemplos de Cada Digito — O 'Padrao Medio'",
        fontsize=14, fontweight="bold"
    )

    for digito in range(10):
        # Pega os primeiros 200 exemplos de cada dígito
        indices = np.where(y_treino == digito)[0][:200]
        exemplos = x_treino[indices]

        # Calcula a MÉDIA de todos os 200 exemplos pixel a pixel
        # Isso revela onde o traço "costuma" aparecer para esse dígito
        media = np.mean(exemplos, axis=0)

        # Posição na grade 2x5
        linha = digito // 5
        coluna = digito % 5
        ax = axes[linha][coluna]

        im = ax.imshow(media, cmap="hot", vmin=0, vmax=255)
        ax.set_title(f"Digito {digito}\n(media de 200)", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print("-> Observe como cada digito tem um 'formato medio' caracteristico!")
    print("   Isso prova que ha um padrao nos dados que a rede pode aprender.")
    print()

    # Mostra também a variação (desvio padrão) — onde os pixels variam mais
    print("Calculando variacao dos pixels entre exemplos do mesmo digito...")
    fig2, axes2 = plt.subplots(2, 5, figsize=(16, 7))
    fig2.suptitle(
        "Variacao dos Pixels (Desvio Padrao) — Onde Cada Pessoa Escreve Diferente",
        fontsize=13, fontweight="bold"
    )

    for digito in range(10):
        indices = np.where(y_treino == digito)[0][:500]
        exemplos = x_treino[indices]
        desvio = np.std(exemplos, axis=0)

        linha = digito // 5
        coluna = digito % 5
        ax = axes2[linha][coluna]

        im = ax.imshow(desvio, cmap="plasma")
        ax.set_title(f"Digito {digito}\n(variacao)", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    print("-> Regioes mais claras = onde as pessoas escrevem de forma mais variada.")
    print("   A rede aprende a ser 'tolerante' com essas variacoes!")


# =============================================================================
# O QUE SÃO "FEATURES" (CARACTERÍSTICAS)
# =============================================================================

def explicar_features():
    """
    No mundo do Machine Learning, "features" (características) são
    as variáveis que usamos para fazer predições.

    No nosso caso:
        Features = os 784 pixels de cada imagem
        Target   = o dígito (0-9) que queremos prever

    A rede neural vai descobrir automaticamente quais combinações
    de pixels são importantes para identificar cada dígito.

    Por exemplo:
        - O "1" tem pixels altos em uma coluna vertical central
        - O "0" tem pixels altos em um anel (oval)
        - O "7" tem pixels altos em uma linha horizontal no topo
          e uma diagonal para baixo

    Nós NÃO programamos essas regras — a rede descobre sozinha!
    """
    print("\n" + "=" * 60)
    print("O QUE SAO 'FEATURES' (CARACTERISTICAS)?")
    print("=" * 60)
    print()
    print("  Features = as informacoes que usamos para fazer a previsao")
    print()
    print("  No MNIST:")
    print("  -> 784 features (um numero por pixel)")
    print("  -> 1 target (qual digito e: 0 a 9)")
    print()
    print("  A rede neural aprende AUTOMATICAMENTE:")
    print("  -> Quais pixels sao importantes para o '1'?")
    print("  -> Quais pixels separam o '3' do '8'?")
    print("  -> Quais combinacoes de pixels indicam cada digito?")
    print()
    print("  Isso e o poder do Deep Learning: nao precisamos programar")
    print("  as regras — a rede descobre sozinha durante o treinamento!")


def resumo_parte2():
    """
    O que aprendemos sobre os dados e o que vem a seguir.
    """
    print()
    print("=" * 60)
    print("RESUMO DA PARTE 2 — O QUE APRENDEMOS:")
    print("=" * 60)
    print()
    print("1. Cada pixel e um numero de 0 (preto) a 255 (branco)")
    print("   -> A rede neural trabalha DIRETAMENTE com esses numeros")
    print()
    print("2. Normalizamos dividindo por 255 -> valores de 0.0 a 1.0")
    print("   -> Estabiliza o treinamento e acelera a convergencia")
    print()
    print("3. Digitos iguais tem pixels parecidos — ha um padrao!")
    print("   -> Isso e o que torna o aprendizado possivel")
    print()
    print("4. Features = os 784 pixels | Target = o digito (0-9)")
    print("   -> A rede aprende a mapear features -> target")
    print()
    print("-" * 60)
    print("PROXIMA PARTE: Preparando os Dados para a Rede Neural")
    print("   -> Normalizar, reformatar, codificar os labels")
    print("   -> Execute: GO1057-Tutorial03-PreparandoOsDados.py")
    print("=" * 60)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Carrega os dados
    x_treino, y_treino, x_teste, y_teste = carregar_dados()

    # Explica o que cada pixel significa
    explicar_pixels(x_treino, y_treino)

    # Explica normalização e normaliza
    x_normalizado = explicar_normalizacao(x_treino, y_treino)

    # Gráfico 1: Heatmap numerado de pixels
    visualizar_heatmap(x_treino, y_treino)

    # O que são features
    explicar_features()

    # Gráfico 2: Overlay de múltiplos exemplos do mesmo dígito
    visualizar_overlay(x_treino, y_treino)

    # Resumo
    resumo_parte2()
