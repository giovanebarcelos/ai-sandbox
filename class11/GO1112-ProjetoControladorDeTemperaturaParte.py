# GO1112-ProjetoControladorDeTemperaturaParte
"""
Controlador de Temperatura Fuzzy - Parte 2: Regras e Simulação
(Continuação do código anterior)
"""

# =============================================================================
# 6. BASE DE REGRAS COMPLETA (15 regras)
# =============================================================================

# BLOCO 1 — BASE DE REGRAS: 15 regras = 5 termos de temp × 3 de umidade.
# Para outro problema: o número de regras = produto do nº de termos de cada entrada.
# Cada regra encapsula o conhecimento especialista do domínio.
# Regras para temperatura muito fria
regra1 = ctrl.Rule(temperatura['muito_fria'] & umidade['seca'], potencia['aquecer_forte'])
regra2 = ctrl.Rule(temperatura['muito_fria'] & umidade['normal'], potencia['aquecer_forte'])
regra3 = ctrl.Rule(temperatura['muito_fria'] & umidade['umida'], potencia['aquecer_forte'])

# Regras para temperatura fria
regra4 = ctrl.Rule(temperatura['fria'] & umidade['seca'], potencia['aquecer_leve'])
regra5 = ctrl.Rule(temperatura['fria'] & umidade['normal'], potencia['aquecer_leve'])
regra6 = ctrl.Rule(temperatura['fria'] & umidade['umida'], potencia['aquecer_forte'])

# Regras para temperatura agradável
regra7 = ctrl.Rule(temperatura['agradavel'] & umidade['seca'], potencia['desligado'])
regra8 = ctrl.Rule(temperatura['agradavel'] & umidade['normal'], potencia['desligado'])
regra9 = ctrl.Rule(temperatura['agradavel'] & umidade['umida'], potencia['resfriar_leve'])

# Regras para temperatura quente
regra10 = ctrl.Rule(temperatura['quente'] & umidade['seca'], potencia['resfriar_leve'])
regra11 = ctrl.Rule(temperatura['quente'] & umidade['normal'], potencia['resfriar_leve'])
regra12 = ctrl.Rule(temperatura['quente'] & umidade['umida'], potencia['resfriar_forte'])

# Regras para temperatura muito quente
regra13 = ctrl.Rule(temperatura['muito_quente'] & umidade['seca'], potencia['resfriar_forte'])
regra14 = ctrl.Rule(temperatura['muito_quente'] & umidade['normal'], potencia['resfriar_forte'])
regra15 = ctrl.Rule(temperatura['muito_quente'] & umidade['umida'], potencia['resfriar_forte'])

# BLOCO 2 — SISTEMA DE CONTROLE: agrupa todas as regras. Para adicionar
# ou remover regras, altere apenas a lista passada para ControlSystem().
# =============================================================================
# 7. CRIAR SISTEMA DE CONTROLE
# =============================================================================

sistema_ctrl = ctrl.ControlSystem([
    regra1, regra2, regra3, regra4, regra5,
    regra6, regra7, regra8, regra9, regra10,
    regra11, regra12, regra13, regra14, regra15
])

controlador = ctrl.ControlSystemSimulation(sistema_ctrl)

# =============================================================================
# 8. FUNÇÃO PARA CONTROLAR TEMPERATURA
# =============================================================================

def controlar_temperatura(temp_atual, umid_atual):
    """
    Calcula a potência necessária para controle de temperatura
    Retorna valor entre -100 (aquece forte) e +100 (resfria forte)
    """
    controlador.input['temperatura'] = temp_atual
    controlador.input['umidade'] = umid_atual
    controlador.compute()
    return controlador.output['potencia']

# =============================================================================
# 9. TESTAR O SISTEMA
# =============================================================================

print("\n╔═══════════════════════════════════════════════════════════════════╗")
print("║          SIMULAÇÃO DO CONTROLADOR DE TEMPERATURA                  ║")
print("╠═══════════════════════════════════════════════════════════════════╣")
print("║ Temp(°C) │ Umidade(%) │ Potência │ Ação                          ║")
print("╠══════════╪════════════╪══════════╪════════════════════════════════╣")

# BLOCO 3 — CENÁRIOS DE TESTE: 6 situações cobrindo extremos e casos normais.
# Para outro problema: defina cenários que exercitem todas as regiões do
# espaço de entrada para validar o comportamento do controlador.
cenarios = [
    (5, 30, "Muito frio, seco"),
    (15, 50, "Frio, normal"),
    (22, 50, "Agradável, normal"),
    (28, 40, "Quente, seco"),
    (35, 80, "Muito quente, úmido"),
    (22, 85, "Agradável, úmido"),
]

for temp, umid, descricao in cenarios:
    pot = controlar_temperatura(temp, umid)
    if pot < -30:
        acao = "Aquecimento forte"
    elif pot < 0:
        acao = "Aquecimento leve"
    elif pot < 30:
        acao = "Desligado/Mínimo"
    elif pot < 60:
        acao = "Resfriamento leve"
    else:
        acao = "Resfriamento forte"

    print(f"║   {temp:2d}     │     {umid:2d}     │  {pot:6.1f}  │ {acao:<30} ║")

print("╚═══════════════════════════════════════════════════════════════════╝")

# =============================================================================
# 10. VISUALIZAR SUPERFÍCIE DE CONTROLE
# =============================================================================

# BLOCO 4 — SUPERFÍCIE DE CONTROLE: mostra a resposta do sistema para
# todas as combinações (temp, umidade). É o 'mapa' completo do controlador.
# Para outro problema: ajuste os ranges para o universo das suas variáveis.
# Gerar grade de valores
temp_range = np.arange(0, 51, 2)
umid_range = np.arange(0, 101, 5)
pot_grid = np.zeros((len(temp_range), len(umid_range)))

for i, t in enumerate(temp_range):
    for j, u in enumerate(umid_range):
        pot_grid[i, j] = controlar_temperatura(t, u)

# Plotar superfície
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

T, U = np.meshgrid(temp_range, umid_range)
ax.plot_surface(T, U, pot_grid.T, cmap='coolwarm', alpha=0.8)

ax.set_xlabel('Temperatura (°C)')
ax.set_ylabel('Umidade (%)')
ax.set_zlabel('Potência')
ax.set_title('Superfície de Controle Fuzzy')

plt.tight_layout()
plt.show()
