# =============================================================================
# Identificador: GO1060-Tutorial06-AvaliacaoEPredicoes
# Aula 10 — MLP: Redes Neurais Multicamadas
# Tutorial Passo a Passo: Reconhecimento de Dígitos Escritos à Mão
# =============================================================================
#
# PARTE 6 DE 7 — AVALIAÇÃO E PREDIÇÕES
#
# A rede foi treinada! Mas o treino não garante que ela funciona bem
# com dados novos. Precisamos AVALIAR com rigor.
#
# Esta parte cobre:
#   1. A diferença fundamental entre treino, validação e teste
#   2. model.evaluate() — acurácia no conjunto de teste
#   3. model.predict() — obtendo as probabilidades de cada classe
#   4. Visualizar predições corretas e erradas
#   5. Analisar onde a rede erra (casos difíceis)
#   6. Matriz de confusão — "mapa de erros" da rede
#   7. Métricas de classificação: precision, recall, F1-score
# =============================================================================

# matplotlib sempre no nível do módulo
import matplotlib
import matplotlib.pyplot as plt

# Configura backend baseado no ambiente
try:
    get_ipython()
    matplotlib.use("module://matplotlib_inline.backend_inline")
except NameError:
    pass

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


# =============================================================================
# PREPARAR TUDO (dados + modelo treinado)
# =============================================================================

def preparar_e_treinar():
    """
    Treina o modelo completo para esta avaliação.
    Em produção, você treinaria uma vez, salvaria e reutilizaria.
    Aqui repetimos para que o tutorial seja auto-contido.
    """
    print("=" * 60)
    print("TUTORIAL AULA 10 — PARTE 6: Avaliacao e Predicoes")
    print("=" * 60)
    print()
    print("Carregando, preparando e treinando... (pode levar alguns minutos)")
    print()

    # Dados
    (x_treino, y_treino), (x_teste, y_teste) = keras.datasets.mnist.load_data()
    x_treino = x_treino.astype("float32") / 255.0
    x_teste  = x_teste.astype("float32")  / 255.0

    # Modelo
    model = keras.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5,
            restore_best_weights=True, verbose=0
        )
    ]

    model.fit(
        x_treino, y_treino,
        epochs=20, batch_size=128,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )

    print("\nModelo treinado com sucesso!")
    return model, x_treino, y_treino, x_teste, y_teste


# =============================================================================
# A DIFERENÇA ENTRE TREINO, VALIDAÇÃO E TESTE
# =============================================================================

def explicar_conjuntos():
    """
    Este conceito é um dos mais importantes em Machine Learning.
    Errar aqui invalida todo o trabalho.

    TREINO (60.000 imagens):
    → A rede APRENDE com esses dados.
    → Os pesos são ajustados baseados nesses exemplos.
    → Métricas aqui são otimistas — a rede "conhece" esses dados.

    VALIDAÇÃO (6.000 imagens, extraídas do treino):
    → A rede NÃO aprende, mas usamos para tomar decisões.
    → Usada para: escolher hiperparâmetros, detectar overfitting.
    → Indiretamente "influencia" o modelo (ex: EarlyStopping).

    TESTE (10.000 imagens — NUNCA vistas):
    → A rede NUNCA viu esses dados durante o treino.
    → Avaliação FINAL, honesta, publicável.
    → Se você usar o teste para ajustar o modelo, ele vira "validação"
      e você precisa de novos dados de teste. Sem atalhos!

    ANALOGIA do vestibular:
    - Treino = estudar com o material de revisão
    - Validação = fazer os simulados para ajustar o estudo
    - Teste = a prova oficial (sem espiar as respostas antes!)
    """
    print("=" * 60)
    print("TREINO vs VALIDACAO vs TESTE — Por que a Diferenca Importa?")
    print("=" * 60)
    print()
    print("TREINO:    A rede APRENDE — pesos sao ajustados")
    print("VALIDACAO: Monitoramento — detecta overfitting")
    print("TESTE:     Avaliacao FINAL — dados que a rede nunca viu!")
    print()
    print("REGRA DE OURO:")
    print("  -> NUNCA use os dados de teste para tomar decisoes")
    print("     sobre o modelo (hiperparametros, arquitetura, etc.)")
    print("  -> Se usar, eles se tornam 'validacao' e voce precisa")
    print("     de um novo conjunto de teste verdadeiramente inedito.")
    print()
    print("Por que isso importa na pratica?")
    print("  -> Um modelo com 99% no treino e 70% no teste e INUTIL")
    print("     (decorou os dados, nao aprendeu nada geral)")
    print("  -> Queremos modelos que generalizem para dados novos!")


# =============================================================================
# AVALIAÇÃO COM model.evaluate()
# =============================================================================

def avaliar_modelo(model, x_teste, y_teste):
    """
    model.evaluate() calcula o loss e as métricas no conjunto de teste.
    Esta é a avaliação "oficial" do nosso modelo.

    Retorna [loss, accuracy] conforme configurado no compile().
    """
    print("\n" + "=" * 60)
    print("AVALIANDO NO CONJUNTO DE TESTE")
    print("=" * 60)
    print()
    print("model.evaluate(x_teste, y_teste):")
    print("  -> Passa pelas 10.000 imagens de teste")
    print("  -> Calcula o loss e a acuracia")
    print("  -> A rede NAO aprende nada aqui (pesos nao mudam)")
    print()

    loss, accuracy = model.evaluate(x_teste, y_teste, verbose=1)

    print()
    print(f"  Loss no teste:     {loss:.4f}")
    print(f"  Acuracia no teste: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print()

    if accuracy >= 0.98:
        print("  -> META ATINGIDA! Acuracia >= 98%")
        print("     Nos Correios, isso significa reconhecer corretamente")
        print(f"     {accuracy*10000:.0f} de cada 10.000 CEPs escritos a mao!")
    else:
        print(f"  -> Acuracia de {accuracy*100:.2f}%")
        print("     Para a meta de 98%, tente mais epocas ou ajustar a arquitetura.")

    return loss, accuracy


# =============================================================================
# PREDIÇÕES COM model.predict()
# =============================================================================

def obter_predicoes(model, x_teste, y_teste):
    """
    model.predict() retorna as probabilidades para cada classe.
    Para cada imagem, obtemos um array de 10 números (probabilidades).

    Exemplo de saída para uma imagem:
    [0.00, 0.00, 0.00, 0.99, 0.00, 0.00, 0.00, 0.01, 0.00, 0.00]
     digito 0  1     2     3     4     5     6     7     8     9
    → Predição: dígito 3 com 99% de confiança!

    np.argmax() pega o índice do maior valor → a predição final.
    """
    print("\n" + "=" * 60)
    print("OBTENDO PREDICOES COM model.predict()")
    print("=" * 60)
    print()

    # predict() retorna probabilidades (array N×10)
    probabilidades = model.predict(x_teste, verbose=0)

    # argmax pega o índice do maior valor = predição final
    predicoes = np.argmax(probabilidades, axis=1)

    print(f"  probabilidades.shape: {probabilidades.shape}")
    print(f"  (10.000 imagens, 10 probabilidades cada)")
    print()
    print("Exemplo — primeira imagem do teste:")
    print(f"  Label real: {y_teste[0]}")
    print("  Probabilidades por digito:")
    for d, p in enumerate(probabilidades[0]):
        barra = "#" * int(p * 40)
        flag = " <-- PREDITO" if d == predicoes[0] else ""
        print(f"    Digito {d}: {p:.4f} ({p*100:5.2f}%) {barra}{flag}")

    print()
    print(f"  Total corretos: {np.sum(predicoes == y_teste):,} de {len(y_teste):,}")
    print(f"  Total errados:  {np.sum(predicoes != y_teste):,} de {len(y_teste):,}")

    return probabilidades, predicoes


# =============================================================================
# VISUALIZAR PREDIÇÕES (CORRETAS E ERRADAS)
# =============================================================================

def visualizar_predicoes(x_teste, y_teste, predicoes, probabilidades):
    """
    Visualização das predições com imagem, label real e predito.
    Inclui acertos (fundo verde) e erros (fundo vermelho) juntos.

    Esta visualização é fundamental para "sentir" como a rede se comporta.
    """
    print("\n" + "=" * 60)
    print("VISUALIZANDO PREDICOES CORRETAS E ERRADAS")
    print("=" * 60)

    # Índices de erros
    erros = np.where(predicoes != y_teste)[0]
    acertos = np.where(predicoes == y_teste)[0]

    print(f"\nTotal de erros: {len(erros)} ({len(erros)/len(y_teste)*100:.2f}%)")
    print("Vamos ver 5 acertos e 5 erros...")

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle(
        "Predicoes da Rede Neural — 5 Acertos (topo) e 5 Erros (fundo)",
        fontsize=14, fontweight="bold"
    )

    # Linha 1: 5 acertos
    for i in range(5):
        idx = acertos[i]
        ax = axes[0][i]
        ax.imshow(x_teste[idx], cmap="gray")

        # Confiança da predição
        confianca = probabilidades[idx][predicoes[idx]] * 100

        ax.set_title(
            f"Real: {y_teste[idx]}  |  Pred: {predicoes[idx]}\n"
            f"Confianca: {confianca:.1f}%",
            fontsize=9, color="darkgreen", fontweight="bold"
        )
        ax.axis("off")

        # Borda verde para acertos
        for spine in ax.spines.values():
            spine.set_edgecolor("green")
            spine.set_linewidth(3)

    # Linha 2: 5 erros
    for i in range(5):
        idx = erros[i]
        ax = axes[1][i]
        ax.imshow(x_teste[idx], cmap="gray")

        confianca = probabilidades[idx][predicoes[idx]] * 100

        ax.set_title(
            f"Real: {y_teste[idx]}  |  Pred: {predicoes[idx]}\n"
            f"Confianca: {confianca:.1f}%",
            fontsize=9, color="darkred", fontweight="bold"
        )
        ax.axis("off")

        # Borda vermelha para erros
        for spine in ax.spines.values():
            spine.set_edgecolor("red")
            spine.set_linewidth(3)

    plt.tight_layout()
    plt.show()

    # Analisa os erros
    print("\nAnalisando os erros...")
    print("Por que a rede erra?")
    print("  1. Escrita ambigua: um '1' que parece '7', um '5' que parece '6'")
    print("  2. Imagem incompleta ou com ruido")
    print("  3. Estilo muito diferente do padrao aprendido")
    print()
    print("Exemplos dos primeiros 10 erros:")
    for i in range(min(10, len(erros))):
        idx = erros[i]
        conf = probabilidades[idx][predicoes[idx]] * 100
        print(f"  Imagem #{idx}: Real={y_teste[idx]} | Pred={predicoes[idx]} | "
              f"Conf={conf:.1f}%")


# =============================================================================
# MATRIZ DE CONFUSÃO
# =============================================================================

def plotar_matriz_confusao(y_teste, predicoes):
    """
    A matriz de confusão é uma tabela N×N onde:
    - Linhas = classes reais
    - Colunas = classes preditas
    - Diagonal principal = acertos
    - Fora da diagonal = erros

    COMO LER:
    Linha 5, coluna 3: a rede confundiu um "5" real com "3"
    Linha 4, coluna 9: a rede confundiu um "4" real com "9"

    As células mais brilhantes fora da diagonal revelam
    as confusões mais frequentes da rede.

    PARES MAIS CONFUNDIDOS NO MNIST (tipicamente):
    - 3 ↔ 5 (similares visualmente)
    - 4 ↔ 9 (compartilham traço vertical)
    - 7 ↔ 1 (ambos com traço vertical dominante)
    """
    print("\n" + "=" * 60)
    print("MATRIZ DE CONFUSAO")
    print("=" * 60)

    cm = confusion_matrix(y_teste, predicoes)

    print("\nComo ler a matriz:")
    print("  Linha   = digito REAL")
    print("  Coluna  = digito PREDITO pela rede")
    print("  Diagonal (\\) = acertos (deve ser alta)")
    print("  Resto = erros (deve ser baixo)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Gráfico 1: Matriz com contagens absolutas ──
    ax1 = axes[0]
    sns.heatmap(
        cm,
        annot=True, fmt="d", cmap="Blues",
        xticklabels=range(10), yticklabels=range(10),
        ax=ax1, linewidths=0.5,
        annot_kws={"size": 10}
    )
    ax1.set_title("Matriz de Confusao\n(valores absolutos)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Predito pela rede", fontsize=11)
    ax1.set_ylabel("Real (verdadeiro)", fontsize=11)

    # ── Gráfico 2: Matriz normalizada (proporção) ──
    ax2 = axes[1]
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm,
        annot=True, fmt=".2f", cmap="YlOrRd",
        xticklabels=range(10), yticklabels=range(10),
        ax=ax2, linewidths=0.5,
        annot_kws={"size": 9}, vmin=0, vmax=1
    )
    ax2.set_title("Matriz de Confusao Normalizada\n(proporcao por classe real)",
                  fontsize=13, fontweight="bold")
    ax2.set_xlabel("Predito pela rede", fontsize=11)
    ax2.set_ylabel("Real (verdadeiro)", fontsize=11)

    plt.suptitle(
        "Onde a Rede Mais Erra? Confusoes por Digito",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.show()

    # Identifica as piores confusões
    print("\nTop 5 pares mais confundidos:")
    erros_cm = cm.copy()
    np.fill_diagonal(erros_cm, 0)  # Zera a diagonal (acertos)
    indices = np.unravel_index(np.argsort(erros_cm.ravel())[-5:], erros_cm.shape)
    for real, pred in zip(indices[0][::-1], indices[1][::-1]):
        n = erros_cm[real, pred]
        if n > 0:
            print(f"  Real={real} foi predito como {pred}: {n} vezes")


# =============================================================================
# RELATÓRIO DE CLASSIFICAÇÃO
# =============================================================================

def mostrar_relatorio_classificacao(y_teste, predicoes):
    """
    O classification_report mostra precision, recall e F1 por classe.

    PRECISION (Precisão):
    "De tudo que a rede disse ser dígito X, quantos eram realmente X?"
    → Mede: quando a rede faz uma afirmação, ela está certa?
    → Alta precisão = poucos falsos positivos

    RECALL (Revocação / Sensibilidade):
    "De todos os dígitos X reais, quantos a rede encontrou?"
    → Mede: a rede consegue encontrar todos os casos reais?
    → Alto recall = poucos falsos negativos

    F1-SCORE:
    → Média harmônica de precision e recall
    → Útil quando as classes são desbalanceadas
    → F1 = 2 × (precision × recall) / (precision + recall)

    SUPORTE:
    → Número de exemplos reais de cada classe no conjunto de teste

    EXEMPLO PRÁTICO:
    Imagine que você procura dígitos "3" em 100 envelopes:
    - 10 são realmente "3", a rede identifica 9 corretamente
      e erra 1 (diz que é "8")
    - A rede também "vê" 2 "3" que eram na verdade "8"
    → Precision = 9/11 ≈ 0.82 (de 11 que disse ser "3", 9 eram)
    → Recall    = 9/10 = 0.90 (de 10 que eram "3", encontrou 9)
    → F1        = 2*(0.82*0.90)/(0.82+0.90) ≈ 0.86
    """
    print("\n" + "=" * 60)
    print("RELATORIO DE CLASSIFICACAO")
    print("=" * 60)
    print()
    print("Metricas por digito:")
    print()
    print(classification_report(
        y_teste, predicoes,
        target_names=[f"Digito {i}" for i in range(10)]
    ))
    print()
    print("DICIONARIO:")
    print("  precision: quando diz X, esta certo X% das vezes")
    print("  recall:    de todos os X reais, encontrou X%")
    print("  f1-score:  media harmonica de precision e recall")
    print("  support:   quantos exemplos reais de cada classe no teste")


def resumo_parte6():
    """Recapitula a avaliação e prepara para a solução completa."""
    print()
    print("=" * 60)
    print("RESUMO DA PARTE 6 — AVALIACAO COMPLETA:")
    print("=" * 60)
    print()
    print("model.evaluate():")
    print("  -> Loss e acuracia no conjunto de teste (nunca visto)")
    print()
    print("model.predict():")
    print("  -> Probabilidades para cada uma das 10 classes")
    print("  -> np.argmax() converte em predicao final")
    print()
    print("Matriz de confusao:")
    print("  -> Mapa visual de onde a rede erra")
    print("  -> Diagonal = acertos | Fora = erros")
    print()
    print("classification_report:")
    print("  -> Precision, recall, F1 por classe")
    print("  -> Identifica quais digitos sao mais dificeis")
    print()
    print("-" * 60)
    print("PROXIMA PARTE: Solucao Completa em um unico arquivo")
    print("   -> Pipeline do zero ao resultado final")
    print("   -> Execute: GO1061-Tutorial07-SolucaoCompleta.py")
    print("=" * 60)


# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Treina o modelo (inclui preparação dos dados)
    model, x_treino, y_treino, x_teste, y_teste = preparar_e_treinar()

    # Explica a diferença entre os conjuntos
    explicar_conjuntos()

    # Avalia no conjunto de teste
    loss, accuracy = avaliar_modelo(model, x_teste, y_teste)

    # Obtém predições
    probabilidades, predicoes = obter_predicoes(model, x_teste, y_teste)

    # Gráfico 1: Grade de predições (acertos + erros)
    visualizar_predicoes(x_teste, y_teste, predicoes, probabilidades)

    # Gráfico 2: Matriz de confusão
    plotar_matriz_confusao(y_teste, predicoes)

    # Relatório de classificação
    mostrar_relatorio_classificacao(y_teste, predicoes)

    # Resumo
    resumo_parte6()
