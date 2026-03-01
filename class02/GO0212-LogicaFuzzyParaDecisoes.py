# GO0212-LógicaFuzzyParaDecisões
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# ═══════════════════════════════════════════════════════════════════
# 1. LÓGICA FUZZY - FUNDAMENTOS
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("LÓGICA FUZZY - CONTROLE DE AR-CONDICIONADO")
print("="*70)

class FuzzySet:
    """
    Conjunto Fuzzy com função de pertinência
    """

    def __init__(self, nome, funcao_pertinencia):
        """
        Args:
            nome: Nome do conjunto (ex: "frio", "quente")
            funcao_pertinencia: Função que retorna grau [0,1]
        """
        self.nome = nome
        self.funcao = funcao_pertinencia

    def pertinencia(self, valor):
        """Calcular grau de pertinência de um valor"""
        return self.funcao(valor)

    def __repr__(self):
        return f"FuzzySet({self.nome})"

def triangular(a, b, c):
    """
    Função de pertinência triangular

    Args:
        a, b, c: Pontos do triângulo (a <= b <= c)

    Returns:
        Função de pertinência
    """
    def pertinencia(x):
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:  # b < x < c
            return (c - x) / (c - b)

    return pertinencia

def trapezoidal(a, b, c, d):
    """
    Função de pertinência trapezoidal

    Args:
        a, b, c, d: Pontos do trapézio (a <= b <= c <= d)
    """
    def pertinencia(x):
        if x <= a or x >= d:
            return 0.0
        elif a < x < b:
            return (x - a) / (b - a)
        elif b <= x <= c:
            return 1.0
        else:  # c < x < d
            return (d - x) / (d - a)

    return pertinencia

# ═══════════════════════════════════════════════════════════════════
# 2. VARIÁVEIS LINGUÍSTICAS (TEMPERATURA E UMIDADE)
# ═══════════════════════════════════════════════════════════════════

print("\n📊 DEFININDO VARIÁVEIS LINGUÍSTICAS...")

# TEMPERATURA (15-35°C)
temperatura_fria = FuzzySet("fria", trapezoidal(10, 10, 18, 22))
temperatura_agradavel = FuzzySet("agradável", triangular(18, 23, 28))
temperatura_quente = FuzzySet("quente", trapezoidal(24, 28, 40, 40))

temperaturas_fuzzy = [temperatura_fria, temperatura_agradavel, temperatura_quente]

print("   ✅ Temperatura: fria, agradável, quente")

# UMIDADE (30-90%)
umidade_baixa = FuzzySet("baixa", trapezoidal(0, 0, 40, 50))
umidade_media = FuzzySet("média", triangular(40, 55, 70))
umidade_alta = FuzzySet("alta", trapezoidal(60, 70, 100, 100))

umidades_fuzzy = [umidade_baixa, umidade_media, umidade_alta]

print("   ✅ Umidade: baixa, média, alta")

# POTÊNCIA DO AR (0-100%)
potencia_desligado = FuzzySet("desligado", trapezoidal(0, 0, 10, 20))
potencia_baixa = FuzzySet("baixa", triangular(10, 25, 40))
potencia_media = FuzzySet("média", triangular(30, 50, 70))
potencia_alta = FuzzySet("alta", trapezoidal(60, 75, 100, 100))

potencias_fuzzy = [potencia_desligado, potencia_baixa, potencia_media, potencia_alta]

print("   ✅ Potência AC: desligado, baixa, média, alta")

# ═══════════════════════════════════════════════════════════════════
# 3. BASE DE REGRAS FUZZY
# ═══════════════════════════════════════════════════════════════════

print("\n📋 BASE DE REGRAS FUZZY:")

regras = [
    # SE temperatura é fria E umidade é qualquer → potência desligado
    {"condicoes": [("temp", "fria"), ("umid", "baixa")], "conclusao": "desligado", "nome": "R1"},
    {"condicoes": [("temp", "fria"), ("umid", "média")], "conclusao": "desligado", "nome": "R2"},
    {"condicoes": [("temp", "fria"), ("umid", "alta")], "conclusao": "desligado", "nome": "R3"},

    # SE temperatura é agradável → potência baixa (ou desligado se umidade baixa)
    {"condicoes": [("temp", "agradável"), ("umid", "baixa")], "conclusao": "desligado", "nome": "R4"},
    {"condicoes": [("temp", "agradável"), ("umid", "média")], "conclusao": "baixa", "nome": "R5"},
    {"condicoes": [("temp", "agradável"), ("umid", "alta")], "conclusao": "baixa", "nome": "R6"},

    # SE temperatura é quente → potência média/alta
    {"condicoes": [("temp", "quente"), ("umid", "baixa")], "conclusao": "média", "nome": "R7"},
    {"condicoes": [("temp", "quente"), ("umid", "média")], "conclusao": "média", "nome": "R8"},
    {"condicoes": [("temp", "quente"), ("umid", "alta")], "conclusao": "alta", "nome": "R9"},
]

for regra in regras:
    conds_str = " AND ".join([f"{v}={c}" for v, c in regra['condicoes']])
    print(f"   {regra['nome']}: IF {conds_str} THEN potencia={regra['conclusao']}")

print(f"\n✅ {len(regras)} regras fuzzy definidas")

# ═══════════════════════════════════════════════════════════════════
# 4. MOTOR DE INFERÊNCIA FUZZY (MAMDANI)
# ═══════════════════════════════════════════════════════════════════

def fuzzificar(valor, conjuntos_fuzzy):
    """
    Fuzzificação: converter valor crisp em graus de pertinência

    Args:
        valor: Valor numérico (ex: 25°C)
        conjuntos_fuzzy: Lista de FuzzySets

    Returns:
        Dict {nome: grau_pertinencia}
    """
    graus = {}
    for conjunto in conjuntos_fuzzy:
        graus[conjunto.nome] = conjunto.pertinencia(valor)
    return graus

def inferir_fuzzy(temp_crisp, umid_crisp, regras):
    """
    Inferência Fuzzy (Mamdani)

    1. Fuzzificar entradas
    2. Avaliar regras (AND = min)
    3. Agregar conclusões (OR = max)
    """
    # 1. Fuzzificar
    temp_fuzzy = fuzzificar(temp_crisp, temperaturas_fuzzy)
    umid_fuzzy = fuzzificar(umid_crisp, umidades_fuzzy)

    print(f"\n🔄 FUZZIFICAÇÃO:")
    print(f"   Temperatura {temp_crisp}°C:")
    for nome, grau in temp_fuzzy.items():
        if grau > 0:
            print(f"      {nome}: {grau:.2f}")

    print(f"   Umidade {umid_crisp}%:")
    for nome, grau in umid_fuzzy.items():
        if grau > 0:
            print(f"      {nome}: {grau:.2f}")

    # 2. Avaliar regras
    print(f"\n🧠 AVALIAÇÃO DAS REGRAS:")
    ativacoes = {pot.nome: [] for pot in potencias_fuzzy}

    for regra in regras:
        # Calcular grau de ativação (AND = min)
        graus_cond = []

        for var, termo in regra['condicoes']:
            if var == "temp":
                graus_cond.append(temp_fuzzy[termo])
            else:  # umid
                graus_cond.append(umid_fuzzy[termo])

        ativacao = min(graus_cond)

        if ativacao > 0:
            ativacoes[regra['conclusao']].append(ativacao)
            print(f"   {regra['nome']}: ativação = {ativacao:.2f} → {regra['conclusao']}")

    # 3. Agregar (OR = max)
    agregado = {}
    for conclusao, lista_ativ in ativacoes.items():
        agregado[conclusao] = max(lista_ativ) if lista_ativ else 0.0

    print(f"\n📊 AGREGAÇÃO:")
    for nome, grau in agregado.items():
        if grau > 0:
            print(f"   {nome}: {grau:.2f}")

    return agregado

def defuzzificar(agregado, metodo='centroide'):
    """
    Defuzzificação: converter fuzzy → crisp

    Métodos:
    - centroide: Centro de área (mais comum)
    - bisector: Divide área em 2
    - mom: Mean of Maximum
    """
    if metodo == 'centroide':
        # Método do centróide (centro de área)
        x = np.linspace(0, 100, 1000)
        y_agregado = np.zeros_like(x)

        # Para cada conjunto fuzzy de saída
        for nome, grau_ativ in agregado.items():
            if grau_ativ > 0:
                # Encontrar conjunto fuzzy correspondente
                conjunto = next((p for p in potencias_fuzzy if p.nome == nome), None)
                if conjunto:
                    # Aplicar grau de ativação (min)
                    y_conjunto = np.array([min(grau_ativ, conjunto.pertinencia(xi)) for xi in x])
                    # Agregar (max)
                    y_agregado = np.maximum(y_agregado, y_conjunto)

        # Calcular centróide
        if y_agregado.sum() > 0:
            centroide = np.sum(x * y_agregado) / np.sum(y_agregado)
            return centroide
        else:
            return 0.0

    elif metodo == 'mom':
        # Mean of Maximum
        max_grau = max(agregado.values())
        if max_grau == 0:
            return 0.0

        # Encontrar conjuntos com grau máximo
        max_conjuntos = [nome for nome, grau in agregado.items() if grau == max_grau]

        # Pegar pontos centrais
        pontos = {
            'desligado': 5,
            'baixa': 25,
            'media': 50,
            'alta': 85
        }

        valores = [pontos[nome] for nome in max_conjuntos]
        return np.mean(valores)

    return 0.0

# ═══════════════════════════════════════════════════════════════════
# 5. TESTAR SISTEMA FUZZY
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("TESTANDO CONTROLADOR FUZZY")
print("="*70)

def testar_controlador(temp, umid):
    """Testar sistema com valores específicos"""
    print(f"\n{'='*70}")
    print(f"CENÁRIO: Temperatura={temp}°C, Umidade={umid}%")
    print(f"{'='*70}")

    # Inferência
    agregado = inferir_fuzzy(temp, umid, regras)

    # Defuzzificação
    potencia_crisp = defuzzificar(agregado, metodo='centroide')

    print(f"\n✅ DEFUZZIFICAÇÃO (Centróide):")
    print(f"   Potência do AC: {potencia_crisp:.1f}%")

    # Interpretação
    if potencia_crisp < 15:
        print(f"   🔵 Decisão: AC DESLIGADO")
    elif potencia_crisp < 40:
        print(f"   🟢 Decisão: AC em BAIXA potência")
    elif potencia_crisp < 70:
        print(f"   🟡 Decisão: AC em MÉDIA potência")
    else:
        print(f"   🔴 Decisão: AC em ALTA potência")

    return potencia_crisp

# Cenários de teste
cenarios = [
    (18, 45),   # Frio e seco → desligado
    (23, 55),   # Agradável e médio → baixa
    (28, 60),   # Quente e médio → média
    (32, 75),   # Muito quente e úmido → alta
    (26, 40),   # Intermediário
]

resultados = []
for temp, umid in cenarios:
    potencia = testar_controlador(temp, umid)
    resultados.append((temp, umid, potencia))

# ═══════════════════════════════════════════════════════════════════
# 6. VISUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

# 1. Funções de pertinência - Temperatura
ax1 = fig.add_subplot(gs[0, 0])
x_temp = np.linspace(15, 35, 200)
for conjunto in temperaturas_fuzzy:
    y = [conjunto.pertinencia(xi) for xi in x_temp]
    ax1.plot(x_temp, y, label=conjunto.nome, linewidth=2)
ax1.set_xlabel('Temperatura (°C)')
ax1.set_ylabel('Grau de Pertinência')
ax1.set_title('Funções de Pertinência - Temperatura')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 1.1])

# 2. Funções de pertinência - Umidade
ax2 = fig.add_subplot(gs[0, 1])
x_umid = np.linspace(30, 90, 200)
for conjunto in umidades_fuzzy:
    y = [conjunto.pertinencia(xi) for xi in x_umid]
    ax2.plot(x_umid, y, label=conjunto.nome, linewidth=2)
ax2.set_xlabel('Umidade (%)')
ax2.set_ylabel('Grau de Pertinência')
ax2.set_title('Funções de Pertinência - Umidade')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.1])

# 3. Funções de pertinência - Potência
ax3 = fig.add_subplot(gs[0, 2])
x_pot = np.linspace(0, 100, 200)
for conjunto in potencias_fuzzy:
    y = [conjunto.pertinencia(xi) for xi in x_pot]
    ax3.plot(x_pot, y, label=conjunto.nome, linewidth=2)
ax3.set_xlabel('Potência (%)')
ax3.set_ylabel('Grau de Pertinência')
ax3.set_title('Funções de Pertinência - Potência (Saída)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 1.1])

# 4. Superfície de controle (3D)
ax4 = fig.add_subplot(gs[1, :], projection='3d')

temp_range = np.linspace(15, 35, 25)
umid_range = np.linspace(30, 90, 25)
T, U = np.meshgrid(temp_range, umid_range)
P = np.zeros_like(T)

for i in range(len(temp_range)):
    for j in range(len(umid_range)):
        agregado = inferir_fuzzy(T[j, i], U[j, i], regras)
        P[j, i] = defuzzificar(agregado, metodo='centroide')

surf = ax4.plot_surface(T, U, P, cmap='viridis', alpha=0.8)
ax4.set_xlabel('Temperatura (°C)')
ax4.set_ylabel('Umidade (%)')
ax4.set_zlabel('Potência AC (%)')
ax4.set_title('Superfície de Controle Fuzzy')
plt.colorbar(surf, ax=ax4, shrink=0.5)

# 5. Resultados dos cenários
ax5 = fig.add_subplot(gs[2, :2])

cenarios_labels = [f"T={t}°C\nU={u}%" for t, u, _ in resultados]
potencias = [p for _, _, p in resultados]
colors = ['blue' if p < 15 else 'green' if p < 40 else 'yellow' if p < 70 else 'red' 
          for p in potencias]

bars = ax5.bar(range(len(resultados)), potencias, color=colors, alpha=0.7, edgecolor='black')
ax5.set_xticks(range(len(resultados)))
ax5.set_xticklabels(cenarios_labels, fontsize=9)
ax5.set_ylabel('Potência AC (%)')
ax5.set_title('Resultados dos Cenários de Teste')
ax5.grid(True, alpha=0.3, axis='y')

# Adicionar valores
for i, (bar, pot) in enumerate(zip(bars, potencias)):
    ax5.text(i, pot + 2, f'{pot:.1f}%', ha='center', fontweight='bold')

# 6. Comparação: Lógica Clássica vs Fuzzy
ax6 = fig.add_subplot(gs[2, 2])

# Lógica clássica (if-else rígido)
def controle_classico(temp, umid):
    if temp < 20:
        return 0
    elif temp < 25:
        return 30
    elif temp < 30:
        return 60
    else:
        return 100

temp_test = np.linspace(15, 35, 100)
pot_fuzzy = []
pot_classica = []

for t in temp_test:
    agregado = inferir_fuzzy(t, 55, regras)  # Umidade fixa em 55%
    pot_fuzzy.append(defuzzificar(agregado))
    pot_classica.append(controle_classico(t, 55))

ax6.plot(temp_test, pot_fuzzy, label='Fuzzy', linewidth=2.5, color='blue')
ax6.plot(temp_test, pot_classica, label='Clássica', linewidth=2.5, 
        linestyle='--', color='red')
ax6.set_xlabel('Temperatura (°C)')
ax6.set_ylabel('Potência (%)')
ax6.set_title('Fuzzy vs Lógica Clássica')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.suptitle('Controlador Fuzzy de Ar-Condicionado', fontsize=14, fontweight='bold')
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 7. RELATÓRIO FINAL
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("RELATÓRIO FINAL - LÓGICA FUZZY")
print("="*70)

print(f"\n🏗️ SISTEMA:")
print(f"   • Método: Mamdani (mais comum)")
print(f"   • Entradas: Temperatura (°C), Umidade (%)")
print(f"   • Saída: Potência AC (0-100%)")
print(f"   • Regras: {len(regras)}")

print(f"\n🔄 PIPELINE:")
print(f"   1. FUZZIFICAÇÃO: Crisp → Fuzzy (graus de pertinência)")
print(f"   2. INFERÊNCIA: Avaliar regras (AND=min, OR=max)")
print(f"   3. AGREGAÇÃO: Combinar conclusões (max)")
print(f"   4. DEFUZZIFICAÇÃO: Fuzzy → Crisp (centróide)")

print(f"\n🎯 APLICAÇÕES REAIS:")
print(f"   • Controle industrial (temperatura, pressão)")
print(f"   • Eletrodomésticos inteligentes")
print(f"   • Automotivo (ABS, câmbio automático)")
print(f"   • Robótica (navegação, tomada de decisão)")
print(f"   • Medicina (sistemas de apoio a diagnóstico)")

print(f"\n💡 VANTAGENS DA LÓGICA FUZZY:")
print(f"   ✅ GRADUAL: Não é binário (0 ou 1)")
print(f"   ✅ INTUITIVO: Regras em linguagem natural")
print(f"   ✅ ROBUSTO: Lida bem com incerteza")
print(f"   ✅ SUAVE: Transições contínuas (não abrutas)")
print(f"   ✅ ESPECIALISTA: Codifica conhecimento humano")

print(f"\n⚙️ FUZZY vs LÓGICA CLÁSSICA:")
print(f"   Clássica: temp > 25 → AC ligado (abrupto)")
print(f"   Fuzzy: temp=25 → 70% quente, 30% agradável (gradual)")

print(f"\n🔬 MÉTODOS DE DEFUZZIFICAÇÃO:")
print(f"   • Centróide: Centro de área (mais usado)")
print(f"   • Bisector: Divide área em 2 partes iguais")
print(f"   • MOM: Mean of Maximum")
print(f"   • LOM/SOM: Largest/Smallest of Maximum")

print("\n✅ CONTROLADOR FUZZY COMPLETO!")
