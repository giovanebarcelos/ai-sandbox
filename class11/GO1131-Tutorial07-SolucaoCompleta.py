# =============================================================================
# Identificador: GO1131-Tutorial07-SolucaoCompleta
# Aula 11 — Logica Fuzzy
# Tutorial Passo a Passo: Sistema de Monitoramento de Qualidade da Agua
# =============================================================================
#
# PARTE 7 DE 7 — SOLUCAO COMPLETA (PIPELINE FINAL)
#
# Este arquivo e o resultado de tudo que aprendemos nas partes 1 a 6.
# E um sistema de monitoramento fuzzy pronto para producao, com:
#
#   - Sistema fuzzy completo (15 regras, 4 entradas, 1 saida)
#   - Funcao de avaliacao robusta com tratamento de erros
#   - Relatorio de qualidade formatado
#   - Historico e tendencia
#   - Recomendacoes de acao baseadas no resultado
#   - Codigo reutilizavel para outros tipos de monitoramento
#
# Para adaptar a outro problema, troque:
#   1. As variaveis (pH, Turbidez, OD, Temperatura)
#   2. Os universos de discurso
#   3. As funcoes de pertinencia
#   4. As regras
#   5. A escala de saida e as acoes recomendadas
#
# Nesta parte voce vai:
#   1. Executar o pipeline completo em um unico arquivo
#   2. Gerar um relatorio profissional de qualidade
#   3. Ver como estender o sistema
# =============================================================================

import matplotlib
import matplotlib.pyplot as plt
try:
    get_ipython()
    matplotlib.use('module://matplotlib_inline.backend_inline')
except NameError:
    pass

import numpy as np
from datetime import datetime

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
except ImportError:
    print("Instale: pip install scikit-fuzzy")
    raise


# =============================================================================
# SISTEMA FUZZY (VERSAO PRODUCAO)
# =============================================================================

class SistemaFuzzyQualidadeAgua:
    """
    Sistema fuzzy completo para monitoramento de qualidade da agua.

    Entradas:
      - pH          : 4.0 a 10.0
      - Turbidez    : 0 a 200 NTU
      - OD          : 0 a 14 mg/L (Oxigenio Dissolvido)
      - Temperatura : 0 a 40 C

    Saida:
      - Indice de Qualidade: 0 a 100
        0-20  : CRITICA  (intervencao urgente)
        20-40 : RUIM     (tratamento avancado)
        40-60 : REGULAR  (tratamento basico)
        60-80 : BOA      (agua adequada)
        80-100: OTIMA    (agua sem restricoes)
    """

    # BLOCO 1 — CLASSIFICAÇÕES: limiares de qualidade baseados em normas de
    # água potável (ex: CONAMA 357). Para outro domínio, substitua estes
    # limiares e ações pelos critérios de aceitação do seu problema.
    # Cada tupla: (indice_min, indice_max, nome, cor_hex, acao_recomendada)
    CLASSIFICACOES = [
        (80, 100, 'OTIMA',   '#26c6da', 'Agua potavel sem restricoes'),
        (60,  80, 'BOA',     '#66bb6a', 'Adequada — monitorar mensalmente'),
        (40,  60, 'REGULAR', '#ffa726', 'Requer tratamento basico'),
        (20,  40, 'RUIM',    '#ff7043', 'Requer tratamento avancado — alertar equipe'),
        ( 0,  20, 'CRITICA', '#f44336', 'Impropia — intervencao URGENTE'),
    ]

    def __init__(self):
        self._simulacao = self._construir()
        self.historico  = []

    # BLOCO 2 — CONSTRUÇÃO DO SISTEMA: encapsula variáveis, MFs e regras.
    # Para outro domínio: este é o único método que precisa ser reescrito —
    # as demais funções (avaliar, relatorio, dashboard) são genéricas.
    def _construir(self):
        ph_u    = np.arange(4.0, 10.1, 0.05)
        turb_u  = np.arange(0.0, 201.0, 1.0)
        od_u    = np.arange(0.0, 14.1, 0.1)
        temp_u  = np.arange(0.0, 40.1, 0.5)
        qual_u  = np.arange(0.0, 100.1, 0.5)

        pH       = ctrl.Antecedent(ph_u,   'pH')
        turbidez = ctrl.Antecedent(turb_u, 'Turbidez')
        od       = ctrl.Antecedent(od_u,   'OD')
        temp     = ctrl.Antecedent(temp_u, 'Temperatura')
        qualidade= ctrl.Consequent(qual_u, 'Qualidade', defuzzify_method='centroid')

        pH['Acido']    = fuzz.trapmf(ph_u,   [4.0, 4.0, 5.5, 6.5])
        pH['Neutro']   = fuzz.trapmf(ph_u,   [6.0, 7.0, 7.5, 8.5])
        pH['Alcalino'] = fuzz.trapmf(ph_u,   [7.5, 9.0,10.0,10.0])

        turbidez['Clara']    = fuzz.trapmf(turb_u, [  0,  0, 20,  50])
        turbidez['Moderada'] = fuzz.trapmf(turb_u, [ 30, 60, 80, 120])
        turbidez['Turva']    = fuzz.trapmf(turb_u, [ 90,140,200, 200])

        od['Baixo'] = fuzz.trapmf(od_u, [0, 0,  3,  5])
        od['Medio'] = fuzz.trapmf(od_u, [4, 6,  7,  9])
        od['Alto']  = fuzz.trapmf(od_u, [8,10, 14, 14])

        temp['Fria']   = fuzz.trapmf(temp_u, [ 0,  0, 10, 18])
        temp['Ideal']  = fuzz.trapmf(temp_u, [15, 20, 24, 28])
        temp['Quente'] = fuzz.trapmf(temp_u, [25, 32, 40, 40])

        qualidade['Critica'] = fuzz.trapmf(qual_u, [ 0,  0, 10, 20])
        qualidade['Ruim']    = fuzz.trapmf(qual_u, [15, 25, 30, 40])
        qualidade['Regular'] = fuzz.trapmf(qual_u, [35, 45, 55, 65])
        qualidade['Boa']     = fuzz.trapmf(qual_u, [60, 70, 75, 82])
        qualidade['Otima']   = fuzz.trapmf(qual_u, [78, 88,100,100])

        regras = [
            ctrl.Rule(pH['Neutro']   & turbidez['Clara']    & od['Alto'],                  qualidade['Otima']),
            ctrl.Rule(pH['Neutro']   & turbidez['Clara']    & od['Medio'] & temp['Ideal'], qualidade['Otima']),
            ctrl.Rule(pH['Neutro']   & turbidez['Moderada'] & od['Alto'],                  qualidade['Boa']),
            ctrl.Rule(pH['Neutro']   & turbidez['Moderada'] & od['Medio'],                 qualidade['Boa']),
            ctrl.Rule(pH['Alcalino'] & turbidez['Clara']    & od['Alto'],                  qualidade['Boa']),
            ctrl.Rule(pH['Acido']    & turbidez['Moderada'] & od['Medio'],                 qualidade['Regular']),
            ctrl.Rule(pH['Neutro']   & turbidez['Moderada'] & od['Baixo'],                 qualidade['Regular']),
            ctrl.Rule(pH['Neutro']   & turbidez['Turva']    & od['Medio'],                 qualidade['Regular']),
            ctrl.Rule(pH['Alcalino'] & turbidez['Moderada'] & od['Medio'],                 qualidade['Regular']),
            ctrl.Rule(pH['Acido']    & turbidez['Turva']    & od['Medio'],                 qualidade['Ruim']),
            ctrl.Rule(pH['Alcalino'] & turbidez['Turva']    & od['Medio'],                 qualidade['Ruim']),
            ctrl.Rule(pH['Neutro']   & turbidez['Turva']    & od['Baixo'],                 qualidade['Ruim']),
            ctrl.Rule(pH['Acido']    & turbidez['Turva']    & od['Baixo'],                 qualidade['Critica']),
            ctrl.Rule(pH['Alcalino'] & turbidez['Turva']    & od['Baixo'],                 qualidade['Critica']),
            ctrl.Rule(temp['Quente'] & od['Baixo'],                                        qualidade['Critica']),
        ]

        sistema_ctrl = ctrl.ControlSystem(regras)
        return ctrl.ControlSystemSimulation(sistema_ctrl)

    def avaliar(self, ph, turbidez, od, temperatura, nome=''):
        """Avalia a qualidade da agua. Retorna dict com resultado completo."""
        ph         = float(np.clip(ph, 4.0, 10.0))
        turbidez   = float(np.clip(turbidez, 0, 200))
        od         = float(np.clip(od, 0, 14))
        temperatura= float(np.clip(temperatura, 0, 40))

        self._simulacao.input['pH']          = ph
        self._simulacao.input['Turbidez']    = turbidez
        self._simulacao.input['OD']          = od
        self._simulacao.input['Temperatura'] = temperatura
        self._simulacao.compute()
        indice = float(self._simulacao.output['Qualidade'])

        classe, cor, acao = self._classificar(indice)
        resultado = {
            'nome': nome,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'entradas': {'ph': ph, 'turbidez': turbidez, 'od': od, 'temp': temperatura},
            'indice': indice,
            'classe': classe,
            'cor': cor,
            'acao': acao,
        }
        self.historico.append(resultado)
        return resultado

    def _classificar(self, indice):
        for minv, maxv, nome, cor, acao in self.CLASSIFICACOES:
            if minv <= indice <= maxv:
                return nome, cor, acao
        return 'INDEFINIDA', '#999', 'Verificar parametros'

    def relatorio(self, resultado):
        """Imprime relatorio formatado de uma avaliacao."""
        e = resultado['entradas']
        sep = "=" * 60
        print(sep)
        print(f"  RELATORIO DE QUALIDADE DA AGUA")
        print(f"  Ponto: {resultado['nome']}")
        print(f"  Data/Hora: {resultado['timestamp']}")
        print(sep)
        print(f"  pH          : {e['ph']:.2f}")
        print(f"  Turbidez    : {e['turbidez']:.1f} NTU")
        print(f"  OD          : {e['od']:.2f} mg/L")
        print(f"  Temperatura : {e['temp']:.1f} C")
        print("-" * 60)
        print(f"  Indice Fuzzy: {resultado['indice']:.1f} / 100")
        print(f"  Qualidade   : {resultado['classe']}")
        print(f"  Acao        : {resultado['acao']}")
        print(sep)
        print()

    def dashboard(self):
        """Gera dashboard visual de todos os resultados no historico."""
        if not self.historico:
            print("Nenhum resultado no historico.")
            return

        n = len(self.historico)
        nomes   = [r['nome'][:22] for r in self.historico]
        indices = [r['indice']    for r in self.historico]
        cores   = [r['cor']       for r in self.historico]
        classes = [r['classe']    for r in self.historico]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Dashboard — Sistema Fuzzy de Qualidade da Agua',
                     fontsize=14, fontweight='bold')

        # Grafico de barras horizontais
        ax1.set_facecolor('#f0f4f8')
        bars = ax1.barh(range(n), indices, color=cores, edgecolor='white',
                        linewidth=1.5, height=0.65)
        for bar, idx, cl in zip(bars, indices, classes):
            ax1.text(idx + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'{idx:.1f}  {cl}', va='center', fontsize=9, fontweight='bold')

        ax1.set_yticks(range(n))
        ax1.set_yticklabels(nomes, fontsize=9)
        ax1.set_xlim(0, 108)
        ax1.set_xlabel('Indice de Qualidade', fontsize=11)
        ax1.set_title('Resultado por Ponto de Coleta', fontsize=12, fontweight='bold')

        # Zonas
        faixas = [(0, 20, '#f44336'), (20, 40, '#ff7043'), (40, 60, '#ffa726'),
                  (60, 80, '#66bb6a'), (80, 100, '#26c6da')]
        for xmin, xmax, cor in faixas:
            ax1.axvspan(xmin, xmax, alpha=0.07, color=cor)
        for lim in [20, 40, 60, 80]:
            ax1.axvline(lim, color='gray', lw=1, ls='--', alpha=0.4)
        ax1.grid(True, axis='x', alpha=0.25)

        # Pizza das classificacoes
        contagem = {}
        for cl in classes:
            contagem[cl] = contagem.get(cl, 0) + 1
        cor_pizza = {'OTIMA': '#26c6da', 'BOA': '#66bb6a', 'REGULAR': '#ffa726',
                     'RUIM': '#ff7043', 'CRITICA': '#f44336'}
        labels_pizza = list(contagem.keys())
        values_pizza = list(contagem.values())
        cores_pizza  = [cor_pizza.get(l, '#999') for l in labels_pizza]

        wedges, texts, autotexts = ax2.pie(
            values_pizza, labels=labels_pizza, colors=cores_pizza,
            autopct='%1.0f%%', startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'},
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_color('white')
        ax2.set_title('Distribuicao das Classificacoes', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.show()


# =============================================================================
# MAIN — PIPELINE COMPLETO
# =============================================================================
if __name__ == '__main__':
    print()
    print("=" * 60)
    print("  TUTORIAL — SISTEMA FUZZY DE QUALIDADE DA AGUA")
    print("  PARTE 7: SOLUCAO COMPLETA (PIPELINE FINAL)")
    print("=" * 60)
    print()
    print("Inicializando sistema fuzzy...")
    sistema = SistemaFuzzyQualidadeAgua()
    print("Sistema pronto. 15 regras carregadas.")
    print()

    # Pontos de coleta para avaliacao
    # BLOCO 3 — PONTOS DE COLETA: 7 cenários cobrem situações reais diversas
    # (preservado, urbano, industrial, turístico, rural). Para outro domínio:
    # substitua pelos casos de uso representativos do seu sistema.
    pontos = [
        ('Manancial A (preservado)', 7.2, 8,   9.5, 20),
        ('Rio B (zona urbana)',      6.8, 55,  6.2, 24),
        ('Lago C (eutrofizado)',     5.5, 130, 2.8, 31),
        ('Reservatorio D',          8.1, 22,  8.0, 18),
        ('Efluente E (industrial)', 4.2, 180, 1.2, 38),
        ('Praia F (turistica)',     7.4, 18,  7.8, 26),
        ('Poco G (rural)',          7.0, 5,   10.2, 19),
    ]

    print("AVALIANDO TODOS OS PONTOS DE COLETA:")
    print()
    for nome, ph, turb, od, temp in pontos:
        resultado = sistema.avaliar(ph, turb, od, temp, nome)
        sistema.relatorio(resultado)

    print("Gerando dashboard...")
    sistema.dashboard()

    # Resumo estatistico
    print("\nRESUMO ESTATISTICO:")
    print("-" * 50)
    indices_all = [r['indice'] for r in sistema.historico]
    print(f"  Pontos avaliados : {len(sistema.historico)}")
    print(f"  Indice medio     : {np.mean(indices_all):.1f}")
    print(f"  Indice min       : {np.min(indices_all):.1f}")
    print(f"  Indice max       : {np.max(indices_all):.1f}")
    print(f"  Desvio padrao    : {np.std(indices_all):.1f}")

    # BLOCO 4 — ALERTAS: limiar 40 separa situações aceitáveis de problemáticas.
    # Para outro domínio: ajuste o limiar de alerta conforme o critério de risco
    # do seu problema (ex: < 60 para sistemas críticos de saúde).
    criticos = [r for r in sistema.historico if r['indice'] < 40]
    if criticos:
        print(f"\n  ALERTAS ({len(criticos)} pontos com qualidade RUIM/CRITICA):")
        for r in criticos:
            print(f"    - {r['nome']}: {r['indice']:.1f} ({r['classe']})")
            print(f"      Acao: {r['acao']}")
    else:
        print("\n  Nenhum ponto com qualidade critica. Monitoramento OK.")

    print()
    print("=" * 60)
    print("  TUTORIAL CONCLUIDO!")
    print()
    print("  O que voce aprendeu:")
    print("  [1] Por que Logica Fuzzy funciona melhor que limites rigidos")
    print("  [2] Funcoes de pertinencia triangular e trapezoidal")
    print("  [3] Variaveis linguisticas e base de regras")
    print("  [4] Inferencia de Mamdani: fuzzificacao -> regras -> defuzz")
    print("  [5] Metodos de defuzzificacao (centroide, MOM, SOM, LOM)")
    print("  [6] API do scikit-fuzzy (Antecedent, Consequent, Rule)")
    print("  [7] Visualizacoes: superficie, sensibilidade, monitoramento")
    print()
    print("  Como adaptar a outro problema:")
    print("  1. Troque as variaveis de entrada e seus universos")
    print("  2. Redefina as MFs conforme o conhecimento do especialista")
    print("  3. Escreva as regras SE-ENTAO")
    print("  4. Ajuste a escala de saida e as acoes recomendadas")
    print("=" * 60)
    print()
