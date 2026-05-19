# GO1108-SistemaDeControle
# Demonstra o uso da biblioteca scikit-fuzzy para construir um sistema de controle fuzzy.
# Instala scikit-fuzzy automaticamente se não estiver disponível (Colab / Jupyter).
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except ImportError:
    import sys, subprocess
    # scikit-fuzzy requer scipy (funções matemáticas) e networkx (grafo de regras)
    subprocess.run([sys.executable, "-m", "pip", "install", "scikit-fuzzy", "scipy", "networkx", "-q"], check=True)
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

if __name__ == "__main__":
    # -----------------------------------------------------------------------
    # BLOCO 1 — UNIVERSOS DE DISCURSO
    # Define a faixa de valores possíveis para cada variável.
    # np.arange(início, fim_exclusivo, passo) — o último valor é fim-passo.
    #   Ex: arange(0, 11, 1) → [0, 1, 2, ..., 10]
    # Passo menor (ex: 0.1) aumenta a precisão da defuzzificação, mas é mais lento.
    # Para outro problema: ajuste os três valores conforme o seu domínio.
    # -----------------------------------------------------------------------
    qualidade_universo = np.arange(0, 11, 1)   # notas de qualidade: 0–10
    servico_universo   = np.arange(0, 11, 1)   # notas de serviço:   0–10
    gorjeta_universo   = np.arange(0, 26, 1)   # gorjeta em %:       0–25

    # -----------------------------------------------------------------------
    # BLOCO 2 — VARIÁVEIS FUZZY
    # Antecedent = variável de entrada (o que o sistema lê).
    # Consequent = variável de saída  (o que o sistema calcula).
    # O segundo argumento ('qualidade', 'servico', 'gorjeta') é o nome da
    # variável — DEVE ser usado exatamente igual em simulacao.input/output
    # no BLOCO 6. Troque esses nomes para o seu domínio.
    # -----------------------------------------------------------------------
    qualidade = ctrl.Antecedent(qualidade_universo, 'qualidade')
    servico   = ctrl.Antecedent(servico_universo,   'servico')
    gorjeta   = ctrl.Consequent(gorjeta_universo,   'gorjeta')

    # -----------------------------------------------------------------------
    # BLOCO 3 — FUNÇÕES DE PERTINÊNCIA
    # Opção A — automf(n): gera n termos triangulares uniformes automaticamente.
    #   n pode ser 3, 5 ou 7. Os termos são sempre nomeados em inglês:
    #   3 termos → 'poor' / 'average' / 'good'
    #   5 termos → 'poor' / 'mediocre' / 'average' / 'decent' / 'good'
    #   Os nomes gerados pelo automf são os que devem ser usados nas regras (BLOCO 4).
    #
    # Opção B — definição manual: use fuzz.trimf / trapmf / gaussmf / sigmf.
    #   fuzz.trimf(universo, [a, b, c]) — triângulo: sobe de a até b, desce de b até c.
    #     [0, 0, 5]  → rampa descendente (máximo na borda esquerda)
    #     [0, 5, 10] → triângulo simétrico com pico em 5
    #     [5, 10, 10]→ rampa ascendente  (máximo na borda direita)
    #   Nomes definidos manualmente (ex: 'ruim', 'media', 'boa') substituem os do automf.
    # -----------------------------------------------------------------------
    servico.automf(3)   # gera: servico['poor'], servico['average'], servico['good']
    gorjeta.automf(3)   # gera: gorjeta['poor'], gorjeta['average'], gorjeta['good']

    # Definição manual para qualidade (sobrescreve o automf):
    qualidade['ruim']  = fuzz.trimf(qualidade.universe, [0, 0, 5])
    qualidade['media'] = fuzz.trimf(qualidade.universe, [0, 5, 10])
    qualidade['boa']   = fuzz.trimf(qualidade.universe, [5, 10, 10])

    # Visualizar as funções de pertinência das três variáveis
    qualidade.view()
    servico.view()
    gorjeta.view()
    plt.show()

    # -----------------------------------------------------------------------
    # BLOCO 4 — REGRAS FUZZY
    # Conectam condições (antecedentes) a conclusões (consequentes).
    # Operadores disponíveis:
    #   |  → OR  (máximo dos graus de pertinência)
    #   &  → AND (mínimo dos graus de pertinência)
    #   ~  → NOT (1 - grau de pertinência)
    # Os termos usados aqui DEVEM existir exatamente como definidos no BLOCO 3.
    #   Ex: qualidade['ruim'] funciona porque 'ruim' foi definido manualmente.
    #       servico['poor']   funciona porque automf(3) gerou o termo 'poor'.
    # Para outro problema: reescreva as regras com os termos do seu domínio.
    # -----------------------------------------------------------------------
    regra1 = ctrl.Rule(qualidade['ruim']  | servico['poor'],   gorjeta['poor'])
    regra2 = ctrl.Rule(servico['average'],                     gorjeta['average'])
    regra3 = ctrl.Rule(servico['good']    | qualidade['boa'],  gorjeta['good'])

    # -----------------------------------------------------------------------
    # BLOCO 5 — SISTEMA DE CONTROLE
    # ControlSystem recebe a lista de regras e monta o grafo de inferência.
    # ControlSystemSimulation é a instância que processa entradas reais.
    # Para outro problema: troque apenas a lista de regras e os nomes.
    # -----------------------------------------------------------------------
    sistema_gorjeta = ctrl.ControlSystem([regra1, regra2, regra3])
    simulacao       = ctrl.ControlSystemSimulation(sistema_gorjeta)

    # -----------------------------------------------------------------------
    # BLOCO 6 — ENTRADA E EXECUÇÃO
    # As chaves de simulacao.input DEVEM ser os nomes dados nos Antecedents
    # do BLOCO 2 ('qualidade', 'servico'). Mesmo vale para simulacao.output.
    # Os valores devem estar dentro do universo definido no BLOCO 1.
    # compute() executa: fuzzificação → inferência → agregação → defuzzificação.
    # -----------------------------------------------------------------------
    simulacao.input['qualidade'] = 6.5   # nota de qualidade (0–10)
    simulacao.input['servico']   = 9.8   # nota de serviço   (0–10)

    simulacao.compute()

    print(f"Gorjeta: {simulacao.output['gorjeta']:.1f}%")

    # Visualiza a saída fuzzy agregada com o centroide (valor defuzzificado) marcado
    gorjeta.view(sim=simulacao)
    plt.show()
