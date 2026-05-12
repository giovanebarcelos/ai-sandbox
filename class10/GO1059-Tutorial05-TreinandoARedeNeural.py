# =============================================================================
# Identificador: GO1059-Tutorial05-TreinandoARedeNeural
# Aula 10 — MLP: Redes Neurais Multicamadas
# Tutorial Passo a Passo: Reconhecimento de Dígitos Escritos à Mão
# =============================================================================
#
# PARTE 5 DE 7 — TREINANDO A REDE NEURAL
#
# Construir a rede é apenas metade do trabalho. Agora vamos TREINÁ-LA.
# Treinar = mostrar os dados repetidamente para que a rede ajuste
# seus 535.818 parâmetros até minimizar os erros.
#
# Esta parte explica:
#   1. compile() — configura COMO a rede vai aprender
#      - optimizer='adam': qual algoritmo ajusta os pesos
#      - loss: como medir o erro
#      - metrics: o que monitorar durante o treino
#   2. fit() — executa o treinamento
#      - epochs: quantas passagens completas pelos dados
#      - batch_size: quantas imagens por atualização de pesos
#      - validation_split: separa dados para verificar generalização
#   3. EarlyStopping — para o treino quando não melhora mais
#   4. Curvas de aprendizado — visualiza o progresso do treino
# =============================================================================

# matplotlib no nível do módulo — sempre
import matplotlib
import matplotlib.pyplot as plt

# Detecta ambiente (Jupyter ou terminal)
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
# PREPARAÇÃO COMPLETA (DADOS + MODELO)
# =============================================================================

def preparar_tudo():
    """
    Prepara dados e modelo prontos para o treinamento.
    Encapsulamos tudo aqui para o código principal ficar limpo.
    """
    print("=" * 60)
    print("TUTORIAL AULA 10 — PARTE 5: Treinando a Rede Neural")
    print("=" * 60)
    print()
    print("Preparando dados e modelo...")

    # Dados
    (x_treino, y_treino), (x_teste, y_teste) = keras.datasets.mnist.load_data()
    x_treino = x_treino.astype("float32") / 255.0
    x_teste  = x_teste.astype("float32")  / 255.0

    # Modelo (mesma arquitetura da Parte 4)
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),
    ])

    print(f"  Dados: {x_treino.shape[0]:,} treino | {x_teste.shape[0]:,} teste")
    print(f"  Modelo: {model.count_params():,} parâmetros")
    print()
    return model, x_treino, y_treino, x_teste, y_teste


# =============================================================================
# EXPLICANDO compile()
# =============================================================================

def explicar_compile(model):
    """
    compile() configura o processo de aprendizado.
    Pense nele como "definir as regras antes do jogo começar".

    Três configurações principais:

    1. optimizer='adam'
    ───────────────────
    O otimizador é o algoritmo que ajusta os pesos após cada batch.
    Adam (Adaptive Moment Estimation) é o mais popular porque:
    - Adapta a taxa de aprendizado para cada parâmetro
    - Combina vantagens do RMSprop e do Momentum
    - Funciona bem na maioria dos casos sem ajuste manual

    ANALOGIA: Adam é como um GPS que recalcula a rota continuamente.
    Se você está indo na direção errada (erro alto), ele sugere
    uma curva grande. Se estiver quase no destino, sugere ajustes finos.

    2. loss='sparse_categorical_crossentropy'
    ─────────────────────────────────────────
    A função de perda (loss) mede o "quão errada" está a rede.
    - "categorical_crossentropy": para classificação com múltiplas classes
    - "sparse_": aceita labels inteiros (3) em vez de one-hot ([0,0,0,1,...])

    Quanto maior o loss, mais errada a rede está.
    O objetivo do treinamento é MINIMIZAR o loss.

    3. metrics=['accuracy']
    ───────────────────────
    Métricas são calculadas para monitoramento, não para otimização.
    Accuracy = proporção de predições corretas (ex: 0.98 = 98% corretos)
    """
    print("=" * 60)
    print("CONFIGURANDO O TREINAMENTO: compile()")
    print("=" * 60)
    print()
    print("Configuracoes:")
    print("  optimizer='adam'")
    print("  -> Algoritmo que ajusta os pesos automaticamente")
    print("  -> Adam: adapta a taxa de aprendizado para cada parametro")
    print("  -> Analogia: GPS que recalcula a rota a cada passo")
    print()
    print("  loss='sparse_categorical_crossentropy'")
    print("  -> Mede o erro da rede")
    print("  -> 'sparse': labels sao numeros inteiros (0,1,2...9)")
    print("  -> 'categorical': problema de classificacao multi-classe")
    print("  -> Objetivo: MINIMIZAR esse valor durante o treino")
    print()
    print("  metrics=['accuracy']")
    print("  -> Monitora a acuracia (% de acertos) a cada epoca")
    print("  -> Nao afeta o treinamento — e so para acompanhar")
    print()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("model.compile(...) executado com sucesso!")
    return model


# =============================================================================
# EXPLICANDO fit() e os PARÂMETROS
# =============================================================================

def explicar_fit():
    """
    fit() é onde o treinamento de fato acontece.

    PARÂMETROS PRINCIPAIS:
    ──────────────────────

    epochs=20
    → Quantas vezes a rede vai "ver" todos os dados de treino.
    → 1 época = passar por todas as 60.000 imagens uma vez.
    → Mais épocas = mais aprendizado (mas risco de overfitting).

    batch_size=128
    → Quantas imagens a rede processa antes de atualizar os pesos.
    → Batch menor = mais atualizações por época (mais "barulhento" mas mais estável)
    → Batch maior = menos atualizações (mais rápido mas pode "pular" mínimos)
    → 128 é um bom equilíbrio para o MNIST.

    ANALOGIA: Pense em revisar um texto longo.
    - batch_size=1: corrige palavra por palavra (lento, nervoso)
    - batch_size=1000: lê o texto inteiro antes de qualquer correção
    - batch_size=128: lê um parágrafo, corrige, lê o próximo (equilíbrio)

    validation_split=0.1
    → Reserva 10% do treino para validação (6.000 imagens).
    → Após cada época, Keras avalia nesses dados sem treinar.
    → Se val_accuracy parar de melhorar → a rede está com overfitting.

    verbose=1
    → Mostra o progresso no terminal (barras de progresso por época).
    → verbose=0: silencioso | verbose=2: uma linha por época.
    """
    print("\n" + "=" * 60)
    print("EXECUTANDO O TREINAMENTO: fit()")
    print("=" * 60)
    print()
    print("Parametros:")
    print("  epochs=20:")
    print("  -> A rede vai ver os 60.000 exemplos 20 vezes")
    print("  -> Mas o EarlyStopping pode parar antes se nao melhorar")
    print()
    print("  batch_size=128:")
    print("  -> Processa 128 imagens, calcula o erro medio, atualiza pesos")
    print("  -> 60.000 / 128 = ~469 atualizacoes por epoca")
    print()
    print("  validation_split=0.1:")
    print("  -> 6.000 imagens reservadas para checar generalizacao")
    print("  -> A rede NAO aprende com esses dados")
    print()
    print("  verbose=1:")
    print("  -> Mostra barra de progresso e metricas por epoca")


# =============================================================================
# EXPLICANDO EarlyStopping
# =============================================================================

def explicar_e_criar_callbacks():
    """
    Callbacks são funções chamadas durante o treinamento.
    EarlyStopping é o callback mais útil para iniciantes.

    EarlyStopping — "para quando parar de melhorar"
    ─────────────────────────────────────────────────
    ANALOGIA: Imagine um aluno estudando para uma prova.
    Ele percebe que, depois de 5 horas de estudo, suas notas
    nos simulados param de melhorar. Continuar estudando
    seria desperdício de tempo (e arriscaria ficar cansado!).

    O EarlyStopping faz isso automaticamente:
    - Monitora uma métrica (ex: val_loss)
    - Se ela não melhorar por `patience` épocas, para o treino
    - Opcionalmente, restaura os melhores pesos encontrados

    PARÂMETROS:
    - monitor='val_loss': monitora o loss na validação
    - patience=5: aguarda 5 épocas sem melhora antes de parar
    - restore_best_weights=True: usa os pesos da melhor época
    """
    print("\n" + "=" * 60)
    print("CALLBACK: EarlyStopping")
    print("=" * 60)
    print()
    print("  EarlyStopping(monitor='val_loss', patience=5, ...")
    print()
    print("  Como funciona:")
    print("  Epoca 1: val_loss = 0.320  <- melhorou! continua")
    print("  Epoca 2: val_loss = 0.285  <- melhorou! continua")
    print("  Epoca 3: val_loss = 0.271  <- melhorou! continua")
    print("  Epoca 4: val_loss = 0.275  <- piorou (contador: 1/5)")
    print("  Epoca 5: val_loss = 0.278  <- piorou (contador: 2/5)")
    print("  Epoca 6: val_loss = 0.274  <- piorou (contador: 3/5)")
    print("  Epoca 7: val_loss = 0.280  <- piorou (contador: 4/5)")
    print("  Epoca 8: val_loss = 0.282  <- piorou (contador: 5/5) PAROU!")
    print()
    print("  restore_best_weights=True:")
    print("  -> Volta para os pesos da Epoca 3 (menor val_loss)")
    print("  -> Garante que usamos o MELHOR modelo, nao o mais recente")

    callback_early = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    # ReduceLROnPlateau — reduz a taxa de aprendizado quando estagna
    callback_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,        # reduz a LR pela metade
        patience=3,
        min_lr=1e-7,
        verbose=1,
    )

    print()
    print("  Bonus — ReduceLROnPlateau:")
    print("  -> Se o loss nao melhorar por 3 epocas, divide a LR por 2")
    print("  -> Permite ajustes mais finos quando esta quase no otimo")

    return [callback_early, callback_lr]


# =============================================================================
# TREINAMENTO
# =============================================================================

def treinar_modelo(model, x_treino, y_treino, callbacks):
    """
    Executa o treinamento.
    O Keras exibirá o progresso automaticamente (verbose=1).

    Após o treino, o objeto `history` contém todo o histórico
    de métricas por época — perfeito para plotar curvas.
    """
    print("\n" + "=" * 60)
    print("INICIANDO O TREINAMENTO...")
    print("=" * 60)
    print()
    print("Aguarde — o treinamento pode levar alguns minutos")
    print("(depende do hardware — GPU e muito mais rapido que CPU)")
    print()

    history = model.fit(
        x_treino,
        y_treino,
        epochs=20,
        batch_size=128,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    print()
    print("Treinamento concluido!")
    return history


# =============================================================================
# VISUALIZANDO AS CURVAS DE APRENDIZADO
# =============================================================================

def plotar_historico(history):
    """
    As curvas de aprendizado são ESSENCIAIS para diagnosticar o treino.

    CURVA DE LOSS:
    - Treino e validação devem cair juntos.
    - Se treino cai mas validação sobe → OVERFITTING (decoreba)
    - Se ambas ficam altas → UNDERFITTING (rede muito simples)

    CURVA DE ACCURACY:
    - Treino e validação devem subir juntos.
    - Diferença grande entre elas → problema de generalização.

    IDEAL: as curvas de treino e validação ficam próximas,
    ambas convergindo para bons valores.
    """
    print("\n" + "=" * 60)
    print("CURVAS DE APRENDIZADO")
    print("=" * 60)

    epocas = range(1, len(history.history["loss"]) + 1)
    melhor_epoca = np.argmin(history.history["val_loss"]) + 1

    print(f"\nMelhor epoca (menor val_loss): {melhor_epoca}")
    print(f"  val_loss final:     {history.history['val_loss'][-1]:.4f}")
    print(f"  val_accuracy final: {history.history['val_accuracy'][-1]:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Gráfico 1: Loss ──
    ax1.plot(epocas, history.history["loss"],     "b-o",  markersize=4,
             label="Treino", linewidth=2)
    ax1.plot(epocas, history.history["val_loss"], "r--o", markersize=4,
             label="Validacao", linewidth=2)
    ax1.axvline(x=melhor_epoca, color="green", linestyle=":", linewidth=1.5,
                alpha=0.7, label=f"Melhor (ep. {melhor_epoca})")
    ax1.set_title("Loss por Epoca\n(quanto menor, melhor)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoca", fontsize=11)
    ax1.set_ylabel("Loss (erro medio)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)

    # Zona de overfitting (quando val_loss > train_loss por muita margem)
    ax1.fill_between(
        epocas,
        history.history["loss"],
        history.history["val_loss"],
        alpha=0.1, color="red", label="Diferenca treino/val"
    )

    # ── Gráfico 2: Accuracy ──
    ax2.plot(epocas, history.history["accuracy"],     "b-o",  markersize=4,
             label="Treino", linewidth=2)
    ax2.plot(epocas, history.history["val_accuracy"], "r--o", markersize=4,
             label="Validacao", linewidth=2)
    ax2.axvline(x=melhor_epoca, color="green", linestyle=":", linewidth=1.5,
                alpha=0.7, label=f"Melhor (ep. {melhor_epoca})")
    ax2.set_title("Acuracia por Epoca\n(quanto maior, melhor)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoca", fontsize=11)
    ax2.set_ylabel("Acuracia (0 a 1)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0.9, 1.01)  # Zoom na região de interesse

    # Linha de referência 98%
    ax2.axhline(y=0.98, color="orange", linestyle="--", alpha=0.7, linewidth=1.5)
    ax2.text(1, 0.981, "Meta: 98%", color="orange", fontsize=9)

    plt.suptitle(
        "Historico de Treinamento — Loss e Acuracia por Epoca",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    # Diagnóstico
    print()
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc   = history.history["val_accuracy"][-1]
    gap = final_train_acc - final_val_acc

    print("DIAGNOSTICO DAS CURVAS:")
    print(f"  Acuracia final treino:    {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"  Acuracia final validacao: {final_val_acc:.4f}   ({final_val_acc*100:.2f}%)")
    print(f"  Diferenca (gap):          {gap:.4f}")

    if gap < 0.005:
        print("  -> Gap pequeno: boa generalizacao! O Dropout esta funcionando.")
    elif gap < 0.02:
        print("  -> Gap moderado: aceitavel para este dataset.")
    else:
        print("  -> Gap grande: possivel overfitting. Aumente o Dropout ou reduza epocas.")


def resumo_parte5():
    """Recapitula o processo de treinamento."""
    print()
    print("=" * 60)
    print("RESUMO DA PARTE 5 — TREINAMENTO CONCLUIDO:")
    print("=" * 60)
    print()
    print("compile() configurou:")
    print("  optimizer: adam (ajusta pesos automaticamente)")
    print("  loss: sparse_categorical_crossentropy (mede o erro)")
    print("  metrics: accuracy (monitora acertos)")
    print()
    print("fit() treinou com:")
    print("  epochs=20 (maximo, EarlyStopping pode parar antes)")
    print("  batch_size=128 (equilibrio entre velocidade e estabilidade)")
    print("  validation_split=0.1 (10% para monitorar generalizacao)")
    print()
    print("Callbacks usados:")
    print("  EarlyStopping: para quando val_loss para de melhorar")
    print("  ReduceLROnPlateau: reduz LR quando estagna")
    print()
    print("-" * 60)
    print("PROXIMA PARTE: Avaliacao e Predicoes")
    print("   -> Testar no conjunto de teste (dados nunca vistos)")
    print("   -> Matriz de confusao, exemplos certos e errados")
    print("   -> Execute: GO1060-Tutorial06-AvaliacaoEPredicoes.py")
    print("=" * 60)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Prepara dados e modelo
    model, x_treino, y_treino, x_teste, y_teste = preparar_tudo()

    # Explica compile()
    model = explicar_compile(model)

    # Explica fit()
    explicar_fit()

    # Explica e cria callbacks
    callbacks = explicar_e_criar_callbacks()

    # Treina o modelo
    history = treinar_modelo(model, x_treino, y_treino, callbacks)

    # Gráfico 1+2: Loss e Accuracy por época (lado a lado)
    plotar_historico(history)

    # Resumo
    resumo_parte5()
