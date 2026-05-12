# =============================================================================
# Identificador: GO1058-Tutorial04-ConstruindoARedeNeural
# Aula 10 — MLP: Redes Neurais Multicamadas
# Tutorial Passo a Passo: Reconhecimento de Dígitos Escritos à Mão
# =============================================================================
#
# PARTE 4 DE 7 — CONSTRUINDO A REDE NEURAL
#
# Chegou a hora de construir nossa rede neural MLP (Multi-Layer Perceptron).
# Antes de escrever o código, vamos entender CADA componente:
#
# Sequential  → "uma fila de camadas, uma depois da outra"
# Flatten     → "transforma a grade 2D em lista 1D"
# Dense       → "camada totalmente conectada — cada neurônio fala com todos"
# ReLU        → "se negativo, vira zero; senão, passa igual"
# Dropout     → "desligar neurônios aleatórios durante o treino"
# Softmax     → "converte em probabilidades que somam 100%"
#
# ANALOGIA GERAL — A Equipe de Especialistas:
# Imagine que para identificar um dígito, você tem uma equipe:
#   - Camada 1 (512 especialistas): detectam traços básicos (curvas, linhas)
#   - Camada 2 (256 especialistas): combinam os traços em partes do dígito
#   - Saída (10 juízes): cada juiz vota a favor do seu dígito favorito
# O dígito com o juiz mais confiante vence!
# =============================================================================

# Importações no nível do módulo — obrigatório
import matplotlib
import matplotlib.pyplot as plt

# Detecta ambiente e ajusta backend
try:
    get_ipython()
    matplotlib.use("module://matplotlib_inline.backend_inline")
except NameError:
    pass

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# =============================================================================
# CARREGANDO E PREPARANDO OS DADOS
# =============================================================================

def preparar_dados():
    """
    Carrega e prepara os dados da mesma forma que a Parte 3.
    Em um projeto real, isso ficaria em um módulo separado chamado por todos.
    Aqui repetimos para que este tutorial seja auto-contido.
    """
    print("=" * 60)
    print("TUTORIAL AULA 10 — PARTE 4: Construindo a Rede Neural")
    print("=" * 60)
    print()

    (x_treino, y_treino), (x_teste, y_teste) = keras.datasets.mnist.load_data()

    # Normalização e achatamento
    x_treino = x_treino.astype("float32") / 255.0
    x_teste  = x_teste.astype("float32")  / 255.0

    print(f"Dados preparados: {x_treino.shape[0]:,} treino | {x_teste.shape[0]:,} teste")
    print("Normalizados: 0.0 a 1.0 | Shape: (N, 28, 28)")
    print()
    return x_treino, y_treino, x_teste, y_teste


# =============================================================================
# EXPLICANDO CADA COMPONENTE DA REDE
# =============================================================================

def explicar_sequential():
    """
    Sequential: "uma fila de camadas, onde a saída de uma
    é a entrada da próxima"

    É o tipo mais simples de modelo no Keras. Funciona como
    uma linha de produção:
        Entrada → Camada1 → Camada2 → ... → Saída

    Limitação: só funciona para arquiteturas simples em linha reta.
    Para arquiteturas mais complexas (múltiplas entradas, ramificações),
    usaríamos a API Funcional do Keras.
    """
    print("=" * 60)
    print("COMPONENTE 1: Sequential")
    print("=" * 60)
    print()
    print("  model = keras.Sequential()  # Cria o 'contêiner' vazio")
    print()
    print("  Fluxo dos dados:")
    print("  Imagem (28x28)")
    print("     |")
    print("  Flatten   <- 'desenrola' em 784 numeros")
    print("     |")
    print("  Dense(512, 'relu')  <- 512 neuronios")
    print("     |")
    print("  Dropout(0.3)  <- desliga 30% aleatoriamente no treino")
    print("     |")
    print("  Dense(256, 'relu')  <- 256 neuronios")
    print("     |")
    print("  Dropout(0.2)  <- desliga 20% aleatoriamente no treino")
    print("     |")
    print("  Dense(10, 'softmax')  <- 10 probabilidades (0-9)")
    print("     |")
    print("  Resposta! (qual digito e)")


def explicar_dense():
    """
    Dense (camada totalmente conectada):
    Cada neurônio desta camada se conecta a TODOS os neurônios
    da camada anterior.

    ANALOGIA: Uma reunião onde TODOS falam com TODOS.
    Camada com 784 entradas e 512 neurônios:
        784 × 512 = 401.408 conexões só nessa camada!

    Cada conexão tem um PESO (um número que a rede aprende).
    Cada neurônio tem um BIAS (um ajuste adicional).
    Total de parâmetros = 784×512 + 512 = 401.920
    """
    print("\n" + "=" * 60)
    print("COMPONENTE 2: Dense (Camada Totalmente Conectada)")
    print("=" * 60)
    print()
    print("  Dense(unidades, activation)")
    print("  - unidades: quantos neuronios nessa camada")
    print("  - activation: qual funcao de ativacao usar")
    print()
    print("  Matematica por dentro:")
    print("  saida = activation(entrada × pesos + bias)")
    print()
    print("  Dense(512):")
    print("  - 784 entradas × 512 neuronios = 401.408 pesos")
    print("  - + 512 bias = 401.920 parametros no total")
    print()
    print("  Por que tantas conexoes?")
    print("  - A rede pode descobrir combinacoes complexas de pixels")
    print("  - Um neuronio pode 'aprender' a detectar uma curva")
    print("  - Outro neuronio pode detectar uma linha vertical")
    print("  - A proxima camada combina essas deteccoes")


def explicar_relu():
    """
    ReLU (Rectified Linear Unit) — Função de Ativação

    FÓRMULA: f(x) = max(0, x)
    - Se x < 0: retorna 0
    - Se x ≥ 0: retorna x (passa igual)

    POR QUÊ PRECISAMOS DE ATIVAÇÃO?
    ─────────────────────────────────
    Sem ativação não-linear, empilhar camadas Dense seria
    matematicamente equivalente a ter UMA camada só.
    (A composição de funções lineares é linear.)

    A ReLU introduz não-linearidade, permitindo que a rede
    aprenda padrões complexos.

    VANTAGENS DA ReLU sobre Sigmoid/Tanh:
    1. Muito rápida de calcular (apenas compara com 0)
    2. Não sofre de "vanishing gradient" para valores positivos
    3. Funciona bem na prática para a maioria dos casos
    """
    print("\n" + "=" * 60)
    print("COMPONENTE 3: ReLU — Funcao de Ativacao")
    print("=" * 60)
    print()
    print("  ReLU(x) = max(0, x)")
    print()
    print("  Exemplos:")
    for x in [-5.0, -1.0, 0.0, 0.5, 1.0, 3.0]:
        saida = max(0, x)
        print(f"  ReLU({x:5.1f}) = {saida:.1f}")
    print()
    print("  Por que nao Sigmoid em camadas ocultas?")
    print("  - Sigmoid 'esmaga' valores em [0,1], causando gradientes minusculos")
    print("  - Em redes profundas, esses gradientes 'desaparecem' (vanishing gradient)")
    print("  - ReLU mantem gradientes fortes para valores positivos")


def explicar_dropout():
    """
    Dropout — Regularização por Desligamento Aleatório

    PROBLEMA QUE RESOLVE: Overfitting (decoreba)
    ─────────────────────────────────────────────
    Quando a rede "decora" os dados de treino em vez de
    aprender padrões gerais, ela vai bem no treino mas
    mal em dados novos.

    ANALOGIA: Imagine que você vai estudar em grupo.
    Se for sempre o mesmo grupo, os integrantes ficam
    dependentes uns dos outros: um sempre resolve
    determinado tipo de problema, outros ficam "preguiçosos".

    O Dropout é como mudar o grupo aleatoriamente a cada sessão.
    Cada neurônio aprende a ser mais independente e robusto,
    porque não pode depender de nenhum colega específico.

    COMO FUNCIONA:
    - Durante o TREINO: desliga aleatoriamente `rate`% dos neurônios
    - Durante a INFERÊNCIA (uso real): todos ligados, pesos × (1-rate)
    """
    print("\n" + "=" * 60)
    print("COMPONENTE 4: Dropout — Regularizacao")
    print("=" * 60)
    print()
    print("  Dropout(rate)")
    print("  - rate: probabilidade de desligar cada neuronio no treino")
    print()
    print("  Dropout(0.3):")
    print("  -> Durante o TREINO: cada neuronio tem 30% de chance de ser desligado")
    print("  -> Durante o TESTE:  todos os neuronios ligados (nenhum desligado)")
    print()
    print("  Camada com 512 neuronios + Dropout(0.3):")
    print("  -> No treino: ~358 neuronios ativos (70% de 512)")
    print("  -> No teste: 512 neuronios ativos (todos)")
    print()
    print("  Por que isso ajuda?")
    print("  -> A rede aprende representacoes redundantes")
    print("  -> Cada caminho aprende independentemente")
    print("  -> Resultado: melhor generalizacao para dados novos")


def explicar_softmax():
    """
    Softmax — Função de Ativação da Camada de Saída

    PROPÓSITO: Converter os valores brutos de saída em probabilidades.

    FÓRMULA: softmax(x_i) = e^(x_i) / Σ e^(x_j)

    PROPRIEDADES:
    - Todas as saídas estão entre 0 e 1
    - A soma de todas as saídas é exatamente 1.0 (100%)
    - O valor mais alto representa o dígito mais provável

    ANALOGIA — Votação com porcentagem:
    Imagine 10 juízes votando em qual dígito é:
    Juiz 0: 2%  | Juiz 1: 1% | Juiz 2: 3% | Juiz 3: 85% | Juiz 4: 1%
    Juiz 5: 2%  | Juiz 6: 2% | Juiz 7: 1% | Juiz 8: 2%  | Juiz 9: 1%
    Total: 100%  -> A rede diz: "É o dígito 3 com 85% de confiança!"
    """
    print("\n" + "=" * 60)
    print("COMPONENTE 5: Softmax — Probabilidades na Saida")
    print("=" * 60)
    print()
    print("  Dense(10, activation='softmax')")
    print("  -> 10 neuronios = 10 classes (digitos 0 a 9)")
    print("  -> Cada neuronio da sua 'confianca' naquele digito")
    print("  -> As confiancas somam 1.0 (100%)")
    print()
    # Demonstração com numpy
    logits = np.array([-1.5, 0.2, -0.8, 2.9, 0.1, -0.3, -0.7, 0.0, -0.4, -0.6])
    exp_logits = np.exp(logits - np.max(logits))  # subtrai max para estabilidade
    softmax_out = exp_logits / exp_logits.sum()
    print("  Exemplo de saida do modelo para uma imagem do digito '3':")
    for d, prob in enumerate(softmax_out):
        barra = "#" * int(prob * 50)
        flag = " <-- PREDITO" if d == np.argmax(softmax_out) else ""
        print(f"  Digito {d}: {prob:.4f} ({prob*100:5.1f}%) {barra}{flag}")
    print()
    print(f"  Predicao final: digito {np.argmax(softmax_out)} "
          f"(confianca: {np.max(softmax_out)*100:.1f}%)")


# =============================================================================
# CONSTRUINDO O MODELO PASSO A PASSO
# =============================================================================

def construir_modelo_passo_a_passo():
    """
    Construímos o modelo adicionando UMA CAMADA de cada vez.
    Isso ajuda a entender o papel de cada componente.

    Arquitetura final:
        Entrada: 784 pixels (28x28 imagem)
        ↓
        Dense(512, relu)  +  Dropout(0.3)  ← Primeira camada oculta
        ↓
        Dense(256, relu)  +  Dropout(0.2)  ← Segunda camada oculta
        ↓
        Dense(10, softmax)                  ← Camada de saída
    """
    print("\n" + "=" * 60)
    print("CONSTRUINDO O MODELO PASSO A PASSO")
    print("=" * 60)
    print()

    # Passo 1: Criar o container Sequential
    print("Passo 1: Criando o container Sequential (a 'caixa' que guarda as camadas)")
    model = keras.Sequential()
    print("  model = keras.Sequential()  ✓")
    print()

    # Passo 2: Camada Flatten
    print("Passo 2: Flatten — transforma 28x28 em 784")
    print("  Por que: a camada Dense precisa de um vetor, nao de uma matriz")
    model.add(layers.Flatten(input_shape=(28, 28)))
    print("  model.add(Flatten(input_shape=(28, 28)))  ✓")
    print()

    # Passo 3: Primeira camada Dense + ReLU
    print("Passo 3: Dense(512, 'relu') — primeira camada oculta")
    print("  512 neuronios: detectam tracos basicos (curvas, linhas, bordas)")
    print("  ReLU: permite que a rede aprenda padroes nao-lineares")
    model.add(layers.Dense(512, activation="relu"))
    print("  model.add(Dense(512, activation='relu'))  ✓")
    print()

    # Passo 4: Dropout 30%
    print("Passo 4: Dropout(0.3) — regularizacao na primeira camada")
    print("  30% dos 512 neuronios sao desligados aleatoriamente no treino")
    model.add(layers.Dropout(0.3))
    print("  model.add(Dropout(0.3))  ✓")
    print()

    # Passo 5: Segunda camada Dense + ReLU
    print("Passo 5: Dense(256, 'relu') — segunda camada oculta")
    print("  256 neuronios: combinam os tracos da camada anterior em partes do digito")
    print("  Ex: curva + linha horizontal = pode ser o '7'")
    model.add(layers.Dense(256, activation="relu"))
    print("  model.add(Dense(256, activation='relu'))  ✓")
    print()

    # Passo 6: Dropout 20%
    print("Passo 6: Dropout(0.2) — regularizacao na segunda camada")
    print("  20% (menor que o anterior, pois a camada ja e menor)")
    model.add(layers.Dropout(0.2))
    print("  model.add(Dropout(0.2))  ✓")
    print()

    # Passo 7: Camada de saída com Softmax
    print("Passo 7: Dense(10, 'softmax') — camada de saida")
    print("  10 neuronios = 10 digitos possiveis (0 a 9)")
    print("  Softmax: cada neuronio da sua 'confianca' naquele digito")
    print("  Todas as confianças somam 1.0 (100%)")
    model.add(layers.Dense(10, activation="softmax"))
    print("  model.add(Dense(10, activation='softmax'))  ✓")
    print()

    print("Modelo construido com sucesso!")
    return model


# =============================================================================
# RESUMO DO MODELO
# =============================================================================

def mostrar_resumo(model):
    """
    model.summary() exibe um resumo completo da arquitetura.

    Colunas do summary():
    - Layer (type): nome e tipo de cada camada
    - Output Shape: formato dos dados após essa camada
    - Param #: quantos parâmetros (pesos+bias) essa camada tem

    "Trainable params" = parâmetros que serão ajustados no treino
    "Non-trainable params" = fixos (ex: parâmetros de BatchNorm congelados)
    """
    print("\n" + "=" * 60)
    print("RESUMO DO MODELO (model.summary())")
    print("=" * 60)
    print()
    print("Colunas explicadas:")
    print("  Layer:        Nome e tipo da camada")
    print("  Output Shape: Formato dos dados apos essa camada")
    print("                None = varia conforme o batch_size")
    print("  Param #:      Pesos + Bias que serao treinados")
    print()
    model.summary()
    print()
    print("Conta dos parametros (verifique):")
    print("  Flatten:       0       <- so reorganiza, nao aprende")
    print("  Dense(512):    784×512 + 512  = 401.920")
    print("  Dropout:       0       <- nao tem parametros")
    print("  Dense(256):    512×256 + 256  = 131.328")
    print("  Dropout:       0       <- nao tem parametros")
    print("  Dense(10):     256×10  + 10   = 2.570")
    print("  TOTAL:                 = 535.818 parametros")


# =============================================================================
# GRÁFICO: COMPARAÇÃO DE FUNÇÕES DE ATIVAÇÃO
# =============================================================================

def plotar_funcoes_ativacao():
    """
    Compara ReLU e Sigmoid visualmente.
    Isso ajuda a entender por que ReLU é preferida em camadas ocultas.
    """
    x = np.linspace(-5, 5, 500)

    # Definindo as funções
    relu    = np.maximum(0, x)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh    = np.tanh(x)

    # Derivadas (gradientes) — importantes para o backpropagation
    relu_grad    = np.where(x > 0, 1.0, 0.0)
    sigmoid_grad = sigmoid * (1 - sigmoid)
    tanh_grad    = 1 - tanh**2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Gráfico 1: As funções ──
    ax1 = axes[0]
    ax1.plot(x, relu,    color="#4fc3f7", linewidth=2.5, label="ReLU  (camadas ocultas)")
    ax1.plot(x, sigmoid, color="#ff7043", linewidth=2.5, label="Sigmoid (saida binaria)")
    ax1.plot(x, tanh,    color="#69f0ae", linewidth=2.5, label="Tanh  (RNN/LSTM)")
    ax1.axhline(y=0, color="gray", linewidth=0.8, alpha=0.5)
    ax1.axvline(x=0, color="gray", linewidth=0.8, alpha=0.5)
    ax1.set_title("Funcoes de Ativacao", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Entrada x", fontsize=11)
    ax1.set_ylabel("Saida f(x)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(-1.5, 3.5)

    # Anotações
    ax1.annotate(
        "ReLU: max(0,x)\nRapida e eficaz!",
        xy=(2, 2), xytext=(1.5, 3.0),
        arrowprops=dict(arrowstyle="->", color="#4fc3f7"),
        fontsize=9, color="#4fc3f7"
    )
    ax1.annotate(
        "Sigmoid:\n'esmaga' em [0,1]",
        xy=(3, 0.95), xytext=(-4, 0.6),
        arrowprops=dict(arrowstyle="->", color="#ff7043"),
        fontsize=9, color="#ff7043"
    )

    # ── Gráfico 2: As derivadas (gradientes) ──
    ax2 = axes[1]
    ax2.plot(x, relu_grad,    color="#4fc3f7", linewidth=2.5, label="Gradiente ReLU")
    ax2.plot(x, sigmoid_grad, color="#ff7043", linewidth=2.5, label="Gradiente Sigmoid")
    ax2.plot(x, tanh_grad,    color="#69f0ae", linewidth=2.5, label="Gradiente Tanh")
    ax2.axhline(y=0, color="gray", linewidth=0.8, alpha=0.5)
    ax2.axvline(x=0, color="gray", linewidth=0.8, alpha=0.5)
    ax2.set_title("Gradientes (como os pesos sao atualizados)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Entrada x", fontsize=11)
    ax2.set_ylabel("Gradiente f'(x)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)

    # Problema da Sigmoid
    ax2.annotate(
        "Gradiente proximo de 0\n= aprende muito lentamente\n(vanishing gradient!)",
        xy=(4.5, 0.01), xytext=(1.5, 0.15),
        arrowprops=dict(arrowstyle="->", color="#ff7043"),
        fontsize=9, color="#ff7043"
    )

    # Vantagem da ReLU
    ax2.annotate(
        "Gradiente constante = 1\npara valores positivos\n(aprende eficientemente!)",
        xy=(3, 1.0), xytext=(-4.5, 0.85),
        arrowprops=dict(arrowstyle="->", color="#4fc3f7"),
        fontsize=9, color="#4fc3f7"
    )

    plt.suptitle(
        "Por que usamos ReLU nas Camadas Ocultas?",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()


def resumo_parte4():
    """Recapitula a construção do modelo."""
    print()
    print("=" * 60)
    print("RESUMO DA PARTE 4 — MODELO CONSTRUIDO:")
    print("=" * 60)
    print()
    print("Arquitetura:")
    print("  Flatten(28x28)  ->  [784]")
    print("  Dense(512, relu) + Dropout(0.3)  ->  [512]")
    print("  Dense(256, relu) + Dropout(0.2)  ->  [256]")
    print("  Dense(10, softmax)               ->  [10]")
    print()
    print("Total: 535.818 parametros treinaveis")
    print()
    print("Conceitos aprendidos:")
    print("  Sequential: container de camadas em linha reta")
    print("  Dense:      cada neuronio conectado a todos os anteriores")
    print("  ReLU:       max(0,x) — funcao de ativacao eficiente")
    print("  Dropout:    desliga neuronios no treino — evita overfitting")
    print("  Softmax:    converte saidas em probabilidades (somam 1.0)")
    print()
    print("-" * 60)
    print("PROXIMA PARTE: Treinando a Rede Neural")
    print("   -> compile(), fit(), callbacks, curvas de aprendizado")
    print("   -> Execute: GO1059-Tutorial05-TreinandoARedeNeural.py")
    print("=" * 60)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Carrega dados
    x_treino, y_treino, x_teste, y_teste = preparar_dados()

    # Explica cada componente
    explicar_sequential()
    explicar_dense()
    explicar_relu()
    explicar_dropout()
    explicar_softmax()

    # Constrói o modelo passo a passo
    model = construir_modelo_passo_a_passo()

    # Mostra o resumo
    mostrar_resumo(model)

    # Gráfico: funções de ativação comparadas
    plotar_funcoes_ativacao()

    # Resumo
    resumo_parte4()
