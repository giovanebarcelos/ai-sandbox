# GO1116-RepetirAteConvergencia
# Algoritmo Genético (GA) para otimização automática dos parâmetros das MFs.
# Problema: definir os parâmetros (a, b, c) de cada MF manualmente é difícil
# quando não há especialista disponível. O GA evolui a população minimizando
# o MSE entre a saída do sistema fuzzy e os dados de referência.
#
# Como funciona: cada 'individual' é uma lista de parâmetros de MFs
# [a1,b1,c1, a2,b2,c2, ...]. A cada geração: seleção, crossover e mutação
# produzem indivíduos melhores até convergir (repetir até convergência).
#
# Exemplo: otimizar os centros/larguras de 2 MFs triangulares para
# aproximar y = sin(x) usando scikit-fuzzy + DEAP.
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Instalação automática de dependências ──────────────────────────────
import subprocess, sys, importlib

def _pip_install(package):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", package],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

for _pkg in ["deap", "scikit-fuzzy", "numpy", "matplotlib"]:
    _import_name = {"scikit-fuzzy": "skfuzzy"}.get(_pkg, _pkg)
    try:
        importlib.import_module(_import_name)
    except ModuleNotFoundError:
        print(f"Instalando {_pkg}...", end=" ", flush=True)
        _pip_install(_pkg)
        print("OK")

# ── 2. Imports ────────────────────────────────────────────────────────────
import random
import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from deap import base, creator, tools, algorithms

# ── 3. Dados de referência: y = sin(x) ───────────────────────────────────
np.random.seed(42)
random.seed(42)

x_universe = np.linspace(0, 2 * np.pi, 200)
y_target   = np.sin(x_universe)

print(f"Problema: aproximar y = sin(x) com 2 MFs triangulares via GA")
print(f"Espaço de busca: centros e larguras das MFs")

# ── 4. Função de avaliação ────────────────────────────────────────────────
# Individual: [c1, w1, c2, w2]  — centro e largura de cada MF triangular
# Saída fuzzy = média ponderada das MFs avaliadas em x

def fuzzy_output(individual, x):
    """Calcula saída do sistema fuzzy dado o indivíduo e vetor x."""
    c1, w1, c2, w2 = individual
    w1 = max(abs(w1), 0.01)  # largura positiva
    w2 = max(abs(w2), 0.01)
    mf1 = fuzz.trimf(x, [c1 - w1, c1, c1 + w1])
    mf2 = fuzz.trimf(x, [c2 - w2, c2, c2 + w2])
    denom = mf1 + mf2 + 1e-9
    # Saída: soma ponderada usando os centros como valores de consequente
    return (mf1 * c1 + mf2 * c2) / denom

def evaluate(individual):
    y_pred = fuzzy_output(individual, x_universe)
    mse = np.mean((y_target - y_pred) ** 2)
    return (mse,)

# ── 5. Configurar DEAP ────────────────────────────────────────────────────
# Evitar redefinição ao reexecutar a célula no Colab
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# Genes: centros no intervalo [0, 2π], larguras em [0.1, π]
toolbox.register("attr_center", random.uniform, 0, 2 * np.pi)
toolbox.register("attr_width",  random.uniform, 0.1, np.pi)

def make_individual():
    return creator.Individual([
        random.uniform(0, 2 * np.pi),  # c1
        random.uniform(0.1, np.pi),    # w1
        random.uniform(0, 2 * np.pi),  # c2
        random.uniform(0.1, np.pi),    # w2
    ])

toolbox.register("individual", make_individual)
toolbox.register("population",  tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate",    evaluate)
toolbox.register("mate",        tools.cxBlend, alpha=0.5)
toolbox.register("mutate",      tools.mutGaussian, mu=0, sigma=0.3, indpb=0.4)
toolbox.register("select",      tools.selTournament, tournsize=3)

# ── 6. Executar GA ────────────────────────────────────────────────────────
POP_SIZE = 100
N_GEN    = 60

population = toolbox.population(n=POP_SIZE)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min",  np.min)
stats.register("mean", np.mean)
stats.register("max",  np.max)

print(f"\nEvoluindo população: {POP_SIZE} indivíduos × {N_GEN} gerações...")
pop, log = algorithms.eaSimple(
    population, toolbox,
    cxpb=0.7, mutpb=0.2,
    ngen=N_GEN,
    stats=stats,
    halloffame=None,
    verbose=False
)
print("Evolução concluída!")

# ── 7. Melhor indivíduo ───────────────────────────────────────────────────
best = tools.selBest(pop, k=1)[0]
best_mse = best.fitness.values[0]
print(f"\nMelhor indivíduo:  c1={best[0]:.3f}  w1={best[1]:.3f}  "
      f"c2={best[2]:.3f}  w2={best[3]:.3f}")
print(f"MSE final: {best_mse:.6f}  |  RMSE: {np.sqrt(best_mse):.6f}")

# ── 8. Visualizações ──────────────────────────────────────────────────────
gen_numbers = [r["gen"]  for r in log]
min_fit     = [r["min"]  for r in log]
mean_fit    = [r["mean"] for r in log]

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle("GA — Otimização de MFs Fuzzy (DEAP)", fontsize=14, fontweight="bold")

# 8a. Curva de convergência
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(gen_numbers, min_fit,  color="steelblue", linewidth=2, label="Melhor (min MSE)")
ax1.plot(gen_numbers, mean_fit, color="tomato",    linewidth=1.5, linestyle="--", label="Média")
ax1.set_title("Convergência do GA")
ax1.set_xlabel("Geração")
ax1.set_ylabel("MSE")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_yscale("log")

# 8b. MFs do melhor indivíduo
ax2 = fig.add_subplot(gs[0, 1])
c1, w1, c2, w2 = best
w1, w2 = max(abs(w1), 0.01), max(abs(w2), 0.01)
mf1 = fuzz.trimf(x_universe, [c1 - w1, c1, c1 + w1])
mf2 = fuzz.trimf(x_universe, [c2 - w2, c2, c2 + w2])
ax2.plot(x_universe, mf1, color="steelblue", linewidth=2, label=f"MF1  c={c1:.2f}  w={w1:.2f}")
ax2.plot(x_universe, mf2, color="tomato",    linewidth=2, label=f"MF2  c={c2:.2f}  w={w2:.2f}")
ax2.set_title("Funções de Pertinência (melhor GA)")
ax2.set_xlabel("x")
ax2.set_ylabel("μ(x)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 1.15)

# 8c. Predição vs alvo (linha)
y_pred_best = fuzzy_output(best, x_universe)
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(x_universe, y_target,    color="steelblue", linewidth=2,   label="Alvo: sin(x)")
ax3.plot(x_universe, y_pred_best, color="tomato",    linewidth=2,
         linestyle="--", label=f"Fuzzy GA  (RMSE={np.sqrt(best_mse):.4f})")
ax3.fill_between(x_universe,
                 y_target, y_pred_best,
                 alpha=0.15, color="orange", label="Erro")
ax3.set_title("Aproximação Fuzzy Otimizada pelo GA vs Alvo")
ax3.set_xlabel("x")
ax3.set_ylabel("y")
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.show()

# 8d. Distribuição da população final (fitness)
fig2, ax4 = plt.subplots(figsize=(8, 4))
final_fits = [ind.fitness.values[0] for ind in pop]
ax4.hist(final_fits, bins=20, color="steelblue", edgecolor="white", alpha=0.85)
ax4.axvline(best_mse, color="red", linestyle="--", linewidth=1.8,
            label=f"Melhor MSE = {best_mse:.5f}")
ax4.set_title("Distribuição de Fitness — População Final")
ax4.set_xlabel("MSE")
ax4.set_ylabel("Frequência")
ax4.legend()
ax4.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nConcluido.")
