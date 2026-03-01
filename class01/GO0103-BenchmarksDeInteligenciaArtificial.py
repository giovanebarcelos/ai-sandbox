# GO0103-BenchmarksDeInteligênciaArtificial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════
# BENCHMARKS DE INTELIGÊNCIA ARTIFICIAL
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("BENCHMARKS DE IA: AVALIAÇÃO DE MODELOS")
print("="*70)

# ═══════════════════════════════════════════════════════════════════
# HISTÓRICO DE BENCHMARKS
# ═══════════════════════════════════════════════════════════════════

benchmarks_historico = {
    "ImageNet": {
        "ano_criacao": 2009,
        "tarefa": "Classificação de Imagens",
        "metrica": "Top-5 Accuracy",
        "baseline_humana": 94.9,
        "marcos": [
            {"ano": 2012, "modelo": "AlexNet", "score": 84.7},
            {"ano": 2014, "modelo": "GoogLeNet", "score": 93.3},
            {"ano": 2015, "modelo": "ResNet", "score": 96.4},  # Superou humanos!
            {"ano": 2021, "modelo": "CoAtNet", "score": 90.88},  # Top-1
        ],
        "status": "Saturado (modelos superam humanos)"
    },
    "GLUE": {
        "ano_criacao": 2018,
        "tarefa": "Natural Language Understanding",
        "metrica": "Média de 9 tarefas",
        "baseline_humana": 87.1,
        "marcos": [
            {"ano": 2018, "modelo": "BERT-Large", "score": 80.5},
            {"ano": 2019, "modelo": "RoBERTa", "score": 88.5},
            {"ano": 2019, "modelo": "T5", "score": 89.9},
        ],
        "status": "Substituído por SuperGLUE"
    },
    "SuperGLUE": {
        "ano_criacao": 2019,
        "tarefa": "NLU (mais difícil)",
        "metrica": "Média de 8 tarefas",
        "baseline_humana": 89.8,
        "marcos": [
            {"ano": 2020, "modelo": "T5-11B", "score": 88.9},
            {"ano": 2021, "modelo": "GPT-3", "score": 71.8},  # Few-shot
            {"ano": 2023, "modelo": "GPT-4", "score": 89.0},  # Chegando ao humano
        ],
        "status": "Ativo"
    },
    "MMLU": {
        "ano_criacao": 2020,
        "tarefa": "Conhecimento Multidisciplinar",
        "metrica": "Acurácia em 57 tópicos",
        "baseline_humana": 89.8,
        "marcos": [
            {"ano": 2021, "modelo": "GPT-3", "score": 43.9},
            {"ano": 2022, "modelo": "PaLM-540B", "score": 69.3},
            {"ano": 2023, "modelo": "GPT-4", "score": 86.4},
            {"ano": 2024, "modelo": "Gemini Ultra", "score": 90.0},  # Superou!
        ],
        "status": "Ativo (desafiador)"
    },
    "HumanEval": {
        "ano_criacao": 2021,
        "tarefa": "Programação (Python)",
        "metrica": "Pass@1 (% correto 1ª tentativa)",
        "baseline_humana": 100.0,
        "marcos": [
            {"ano": 2021, "modelo": "Codex (GPT-3)", "score": 28.8},
            {"ano": 2023, "modelo": "GPT-4", "score": 67.0},
            {"ano": 2024, "modelo": "Claude 3 Opus", "score": 84.9},
        ],
        "status": "Ativo"
    },
}

# ═══════════════════════════════════════════════════════════════════
# ANÁLISE TEXTUAL
# ═══════════════════════════════════════════════════════════════════

print("\n📊 BENCHMARKS PRINCIPAIS DE IA:\n")

for nome, info in benchmarks_historico.items():
    print(f"{'='*70}")
    print(f"🎯 {nome} ({info['ano_criacao']})")
    print(f"{'='*70}")
    print(f"   Tarefa: {info['tarefa']}")
    print(f"   Métrica: {info['metrica']}")
    print(f"   Baseline Humana: {info['baseline_humana']}%")
    print(f"   Status: {info['status']}")

    print(f"\n   📈 Evolução:")
    for marco in info['marcos']:
        print(f"      {marco['ano']}: {marco['modelo']} → {marco['score']}%")

    # Verificar se superou humanos
    ultimo_score = info['marcos'][-1]['score']
    if ultimo_score >= info['baseline_humana']:
        print(f"   ✅ IA SUPEROU HUMANOS! ({ultimo_score}% vs {info['baseline_humana']}%)")
    else:
        gap = info['baseline_humana'] - ultimo_score
        print(f"   ⏳ Gap para humanos: {gap:.1f} pontos percentuais")

    print()

# ═══════════════════════════════════════════════════════════════════
# VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════

print("\n📊 GERANDO VISUALIZAÇÕES...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Evolução do ImageNet
ax1 = axes[0, 0]
imagenet_data = benchmarks_historico["ImageNet"]["marcos"]
anos_in = [m["ano"] for m in imagenet_data]
scores_in = [m["score"] for m in imagenet_data]
modelos_in = [m["modelo"] for m in imagenet_data]

ax1.plot(anos_in, scores_in, 'o-', linewidth=2.5, markersize=10, color='steelblue', label='IA')
ax1.axhline(y=94.9, color='red', linestyle='--', linewidth=2, label='Humano (94.9%)')

for ano, score, modelo in zip(anos_in, scores_in, modelos_in):
    ax1.annotate(f"{modelo}\n{score}%", xy=(ano, score), 
                xytext=(0, 10), textcoords='offset points',
                fontsize=8, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7))

ax1.set_xlabel("Ano", fontsize=11)
ax1.set_ylabel("Top-5 Accuracy (%)", fontsize=11)
ax1.set_title("ImageNet: IA Superou Humanos em 2015", fontsize=12, fontweight='bold')
ax1.set_ylim(80, 100)
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Comparação de Benchmarks NLP
ax2 = axes[0, 1]
benchmarks_nlp = ["GLUE", "SuperGLUE", "MMLU"]
baselines = [87.1, 89.8, 89.8]
ultimos_ia = [89.9, 89.0, 90.0]  # Melhores scores de IA

x_pos = np.arange(len(benchmarks_nlp))
width = 0.35

bars1 = ax2.bar(x_pos - width/2, baselines, width, label='Humano', 
               color='coral', alpha=0.7, edgecolor='black')
bars2 = ax2.bar(x_pos + width/2, ultimos_ia, width, label='Melhor IA (2023-2024)', 
               color='lightgreen', alpha=0.7, edgecolor='black')

ax2.set_ylabel("Score (%)", fontsize=11)
ax2.set_title("Benchmarks NLP: IA vs Humano", fontsize=12, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(benchmarks_nlp)
ax2.set_ylim(80, 95)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Adicionar valores nas barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# 3. Evolução MMLU (conhecimento)
ax3 = axes[1, 0]
mmlu_data = benchmarks_historico["MMLU"]["marcos"]
anos_mm = [m["ano"] for m in mmlu_data]
scores_mm = [m["score"] for m in mmlu_data]
modelos_mm = [m["modelo"] for m in mmlu_data]

ax3.plot(anos_mm, scores_mm, 'o-', linewidth=2.5, markersize=10, color='purple', label='IA')
ax3.axhline(y=89.8, color='red', linestyle='--', linewidth=2, label='Humano (89.8%)')

for ano, score, modelo in zip(anos_mm, scores_mm, modelos_mm):
    ax3.annotate(f"{modelo}\n{score}%", xy=(ano, score), 
                xytext=(0, -20 if ano == 2021 else 10), textcoords='offset points',
                fontsize=8, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7))

ax3.set_xlabel("Ano", fontsize=11)
ax3.set_ylabel("Acurácia (%)", fontsize=11)
ax3.set_title("MMLU: IA Superou Humanos em 2024", fontsize=12, fontweight='bold')
ax3.set_ylim(40, 95)
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. HumanEval (Programação)
ax4 = axes[1, 1]
he_data = benchmarks_historico["HumanEval"]["marcos"]
modelos_he = [m["modelo"] for m in he_data]
scores_he = [m["score"] for m in he_data]

bars = ax4.barh(modelos_he, scores_he, color=['skyblue', 'lightgreen', 'gold'], 
               alpha=0.7, edgecolor='black')
ax4.axvline(x=100, color='red', linestyle='--', linewidth=2, label='Humano (100%)')

ax4.set_xlabel("Pass@1 (%)", fontsize=11)
ax4.set_title("HumanEval: Programação Python", fontsize=12, fontweight='bold')
ax4.set_xlim(0, 110)
ax4.legend()
ax4.grid(True, alpha=0.3, axis='x')

# Adicionar valores nas barras
for bar, score in zip(bars, scores_he):
    width = bar.get_width()
    ax4.text(width + 2, bar.get_y() + bar.get_height()/2.,
            f'{score}%', ha='left', va='center', fontsize=10, fontweight='bold')

plt.suptitle("Benchmarks de IA: Evolução e Comparação com Humanos", 
            fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# ANÁLISE E DISCUSSÃO
# ═══════════════════════════════════════════════════════════════════

print("\n💡 LIÇÕES DOS BENCHMARKS:")
print("="*70)

print("\n1️⃣ SATURAÇÃO DE BENCHMARKS:")
print("   • ImageNet: IA superou humanos em 2015 → benchmark saturado")
print("   • GLUE: Saturado em 2019 → criaram SuperGLUE")
print("   • Ciclo: Benchmark criado → IA domina → novo benchmark mais difícil")

print("\n2️⃣ PROGRESSO ACELERADO:")
print("   • MMLU: 43.9% (2021) → 90% (2024) em apenas 3 anos!")
print("   • GPT-3 (2020) → GPT-4 (2023): Salto gigante")
print("   • Lei de Scaling: Mais parâmetros + mais dados = melhor performance")

print("\n3️⃣ LIMITAÇÕES DOS BENCHMARKS:")
print("   • Overfitting: Modelos podem memorizar benchmarks públicos")
print("   • Não capturam raciocínio real: Múltipla escolha ≠ Pensamento")
print("   • Faltam: Senso comum, criatividade, ética")

print("\n4️⃣ NOVOS BENCHMARKS (2024+):")
print("   • BIG-Bench (> 200 tarefas)")
print("   • HELM (Holistic Evaluation)")
print("   • Chatbot Arena (humanos votam em conversas)")
print("   • ARC-AGI (François Chollet) - raciocínio abstrato")

print("\n🎯 QUANDO IA SERÁ AGI?")
print("   Critérios propostos:")
print("   ✓ Superar humanos em TODOS os benchmarks cognitivos")
print("   ✓ Aprender novas tarefas com poucos exemplos (few-shot)")
print("   ✓ Raciocínio abstrato e transferência de conhecimento")
print("   ✓ Consciência (?) - problema filosófico em aberto")

print("\n   Estado atual (2026):")
print("   • IA supera humanos em tarefas específicas (ANI)")
print("   • AGI ainda distante: falta raciocínio causal, senso comum")
print("   • Estimativas variam: 5-50 anos")

print("\n✅ ANÁLISE DE BENCHMARKS COMPLETA!")
