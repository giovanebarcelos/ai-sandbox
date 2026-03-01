# GO0211-PrologSimplesEmPython
from collections import defaultdict
import re

# ═══════════════════════════════════════════════════════════════════
# 1. MINI-PROLOG EM PYTHON
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("MINI-PROLOG - RACIOCÍNIO LÓGICO SOBRE RELAÇÕES")
print("="*70)

class MiniProlog:
    """
    Implementação simplificada de Prolog em Python

    Suporta:
    - Fatos: pai(joao, maria)
    - Regras: avo(X, Y) :- pai(X, Z), pai(Z, Y)
    - Consultas: ?- avo(joao, X)
    """

    def __init__(self):
        self.fatos = []  # Lista de fatos
        self.regras = []  # Lista de regras

    def adicionar_fato(self, predicado, *args):
        """Adicionar fato à base"""
        fato = (predicado, tuple(args))
        self.fatos.append(fato)
        print(f"   ✅ Fato: {predicado}({', '.join(args)})")

    def adicionar_regra(self, cabeca, corpo):
        """
        Adicionar regra à base

        Args:
            cabeca: (predicado, (args...))
            corpo: lista de (predicado, (args...))
        """
        regra = {'cabeca': cabeca, 'corpo': corpo}
        self.regras.append(regra)

        # Formatar para exibição
        cabeca_str = f"{cabeca[0]}({', '.join(cabeca[1])})"
        corpo_str = ", ".join([f"{p}({', '.join(args)})" for p, args in corpo])
        print(f"   📋 Regra: {cabeca_str} :- {corpo_str}")

    def unificar(self, termo1, termo2, substituicoes):
        """
        Unificação: tentar fazer match de dois termos

        Args:
            termo1, termo2: Tuplas (predicado, (args...))
            substituicoes: Dict de variáveis já substituídas

        Returns:
            Nova dict de substituições ou None se não unifica
        """
        pred1, args1 = termo1
        pred2, args2 = termo2

        # Predicados diferentes
        if pred1 != pred2:
            return None

        # Número diferente de argumentos
        if len(args1) != len(args2):
            return None

        # Copiar substituições
        novas_subs = substituicoes.copy()

        # Tentar unificar cada argumento
        for a1, a2 in zip(args1, args2):
            # Aplicar substituições existentes
            if a1 in novas_subs:
                a1 = novas_subs[a1]
            if a2 in novas_subs:
                a2 = novas_subs[a2]

            # Ambos são variáveis
            if self.e_variavel(a1) and self.e_variavel(a2):
                if a1 != a2:
                    novas_subs[a1] = a2

            # a1 é variável
            elif self.e_variavel(a1):
                novas_subs[a1] = a2

            # a2 é variável
            elif self.e_variavel(a2):
                novas_subs[a2] = a1

            # Ambos são constantes
            elif a1 != a2:
                return None  # Não unifica

        return novas_subs

    def e_variavel(self, termo):
        """Verificar se termo é variável (começa com maiúscula)"""
        return isinstance(termo, str) and termo[0].isupper()

    def aplicar_substituicoes(self, termo, substituicoes):
        """Aplicar substituições a um termo"""
        predicado, args = termo
        novos_args = tuple(substituicoes.get(a, a) for a in args)
        return (predicado, novos_args)

    def consultar(self, objetivo, substituicoes=None, profundidade=0, max_prof=10):
        """
        Consultar base de conhecimento (com backtracking)

        Args:
            objetivo: (predicado, (args...))
            substituicoes: Dict de substituições
            profundidade: Controle de recursão

        Yields:
            Dict de substituições para cada solução
        """
        if substituicoes is None:
            substituicoes = {}

        if profundidade > max_prof:
            return

        # Aplicar substituições ao objetivo
        objetivo = self.aplicar_substituicoes(objetivo, substituicoes)

        # Tentar unificar com fatos
        for fato in self.fatos:
            novas_subs = self.unificar(objetivo, fato, substituicoes)
            if novas_subs is not None:
                yield novas_subs

        # Tentar unificar com regras
        for regra in self.regras:
            # Renomear variáveis da regra (evitar conflitos)
            regra_renomeada = self.renomear_variaveis(regra, profundidade)

            # Tentar unificar objetivo com cabeça da regra
            novas_subs = self.unificar(objetivo, regra_renomeada['cabeca'], substituicoes)

            if novas_subs is not None:
                # Resolver corpo da regra
                for subs_final in self.resolver_corpo(regra_renomeada['corpo'], novas_subs, profundidade + 1):
                    yield subs_final

    def renomear_variaveis(self, regra, sufixo):
        """Renomear variáveis de uma regra para evitar conflitos"""
        cabeca = regra['cabeca']
        corpo = regra['corpo']

        # Mapear variáveis antigas para novas
        variaveis = set()

        # Coletar variáveis da cabeça
        for arg in cabeca[1]:
            if self.e_variavel(arg):
                variaveis.add(arg)

        # Coletar variáveis do corpo
        for termo in corpo:
            for arg in termo[1]:
                if self.e_variavel(arg):
                    variaveis.add(arg)

        # Criar mapeamento
        mapeamento = {var: f"{var}_{sufixo}" for var in variaveis}

        # Renomear cabeça
        nova_cabeca = (cabeca[0], tuple(mapeamento.get(a, a) for a in cabeca[1]))

        # Renomear corpo
        novo_corpo = []
        for pred, args in corpo:
            novos_args = tuple(mapeamento.get(a, a) for a in args)
            novo_corpo.append((pred, novos_args))

        return {'cabeca': nova_cabeca, 'corpo': novo_corpo}

    def resolver_corpo(self, corpo, substituicoes, profundidade):
        """
        Resolver corpo de uma regra (lista de objetivos)
        """
        if not corpo:
            # Corpo vazio = sucesso
            yield substituicoes
            return

        # Pegar primeiro objetivo
        primeiro = corpo[0]
        resto = corpo[1:]

        # Resolver primeiro objetivo
        for subs in self.consultar(primeiro, substituicoes, profundidade):
            # Resolver resto do corpo
            for subs_final in self.resolver_corpo(resto, subs, profundidade):
                yield subs_final

    def consultar_e_mostrar(self, objetivo_str):
        """
        Consultar e mostrar resultados

        Args:
            objetivo_str: String como "pai(joao, X)"
        """
        # Parse do objetivo
        match = re.match(r'(\w+)\((.*)\)', objetivo_str)
        if not match:
            print(f"❌ Formato inválido: {objetivo_str}")
            return

        predicado = match.group(1)
        args_str = match.group(2)
        args = tuple(a.strip() for a in args_str.split(','))

        objetivo = (predicado, args)

        print(f"\n🔍 CONSULTA: {objetivo_str}")
        print("="*50)

        # Executar consulta
        solucoes = list(self.consultar(objetivo))

        if not solucoes:
            print("   ❌ Nenhuma solução encontrada")
            return

        print(f"   ✅ {len(solucoes)} solução(ões) encontrada(s):\n")

        # Mostrar soluções
        for i, subs in enumerate(solucoes, 1):
            # Filtrar apenas variáveis do objetivo
            variaveis_objetivo = [a for a in args if a[0].isupper()]

            if not variaveis_objetivo:
                print(f"   {i}. Verdadeiro")
            else:
                valores = []
                for var in variaveis_objetivo:
                    valor = subs.get(var, var)
                    valores.append(f"{var} = {valor}")

                print(f"   {i}. {', '.join(valores)}")

# ═══════════════════════════════════════════════════════════════════
# 2. BASE DE CONHECIMENTO: ÁRVORE GENEALÓGICA
# ═══════════════════════════════════════════════════════════════════

prolog = MiniProlog()

print("\n📚 CONSTRUINDO BASE DE CONHECIMENTO (FAMÍLIA)...")
print("\n👨‍👩‍👧‍👦 FATOS: Relações Diretas")

# Fatos: pai(Pai, Filho)
prolog.adicionar_fato("pai", "joao", "maria")
prolog.adicionar_fato("pai", "joao", "pedro")
prolog.adicionar_fato("pai", "pedro", "ana")
prolog.adicionar_fato("pai", "pedro", "carlos")
prolog.adicionar_fato("pai", "jose", "joao")
prolog.adicionar_fato("pai", "jose", "paulo")

# Fatos: mae(Mae, Filho)
prolog.adicionar_fato("mae", "clara", "maria")
prolog.adicionar_fato("mae", "clara", "pedro")
prolog.adicionar_fato("mae", "maria", "ana")
prolog.adicionar_fato("mae", "maria", "carlos")
prolog.adicionar_fato("mae", "lucia", "joao")
prolog.adicionar_fato("mae", "lucia", "paulo")

# Fatos: homem/mulher
prolog.adicionar_fato("homem", "joao")
prolog.adicionar_fato("homem", "pedro")
prolog.adicionar_fato("homem", "jose")
prolog.adicionar_fato("homem", "carlos")
prolog.adicionar_fato("homem", "paulo")

prolog.adicionar_fato("mulher", "maria")
prolog.adicionar_fato("mulher", "clara")
prolog.adicionar_fato("mulher", "ana")
prolog.adicionar_fato("mulher", "lucia")

print(f"\n✅ Base de fatos: {len(prolog.fatos)} fatos")

# ═══════════════════════════════════════════════════════════════════
# 3. REGRAS: Relações Derivadas
# ═══════════════════════════════════════════════════════════════════

print("\n📋 REGRAS: Relações Derivadas")

# Regra: avo(X, Y) :- pai(X, Z), pai(Z, Y)
prolog.adicionar_regra(
    cabeca=("avo", ("X", "Y")),
    corpo=[("pai", ("X", "Z")), ("pai", ("Z", "Y"))]
)

# Regra: avo(X, Y) :- pai(X, Z), mae(Z, Y)
prolog.adicionar_regra(
    cabeca=("avo", ("X", "Y")),
    corpo=[("pai", ("X", "Z")), ("mae", ("Z", "Y"))]
)

# Regra: avó(X, Y) :- mae(X, Z), pai(Z, Y)
prolog.adicionar_regra(
    cabeca=("avo", ("X", "Y")),
    corpo=[("mae", ("X", "Z")), ("pai", ("Z", "Y"))]
)

# Regra: avó(X, Y) :- mae(X, Z), mae(Z, Y)
prolog.adicionar_regra(
    cabeca=("avo", ("X", "Y")),
    corpo=[("mae", ("X", "Z")), ("mae", ("Z", "Y"))]
)

# Regra: irmao(X, Y) :- pai(Z, X), pai(Z, Y), X != Y
prolog.adicionar_regra(
    cabeca=("irmao", ("X", "Y")),
    corpo=[("pai", ("Z", "X")), ("pai", ("Z", "Y"))]
)

# Regra: tio(X, Y) :- irmao(X, Z), pai(Z, Y)
prolog.adicionar_regra(
    cabeca=("tio", ("X", "Y")),
    corpo=[("irmao", ("X", "Z")), ("pai", ("Z", "Y"))]
)

# Regra: tio(X, Y) :- irmao(X, Z), mae(Z, Y)
prolog.adicionar_regra(
    cabeca=("tio", ("X", "Y")),
    corpo=[("irmao", ("X", "Z")), ("mae", ("Z", "Y"))]
)

# Regra: ancestral(X, Y) :- pai(X, Y)
prolog.adicionar_regra(
    cabeca=("ancestral", ("X", "Y")),
    corpo=[("pai", ("X", "Y"))]
)

# Regra: ancestral(X, Y) :- mae(X, Y)
prolog.adicionar_regra(
    cabeca=("ancestral", ("X", "Y")),
    corpo=[("mae", ("X", "Y"))]
)

# Regra: ancestral(X, Y) :- pai(X, Z), ancestral(Z, Y) [RECURSIVA]
prolog.adicionar_regra(
    cabeca=("ancestral", ("X", "Y")),
    corpo=[("pai", ("X", "Z")), ("ancestral", ("Z", "Y"))]
)

# Regra: ancestral(X, Y) :- mae(X, Z), ancestral(Z, Y) [RECURSIVA]
prolog.adicionar_regra(
    cabeca=("ancestral", ("X", "Y")),
    corpo=[("mae", ("X", "Z")), ("ancestral", ("Z", "Y"))]
)

print(f"\n✅ Base de regras: {len(prolog.regras)} regras")

# ═══════════════════════════════════════════════════════════════════
# 4. CONSULTAS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("EXECUTANDO CONSULTAS")
print("="*70)

# Consulta 1: Quem é pai de maria?
prolog.consultar_e_mostrar("pai(X, maria)")

# Consulta 2: Quem são os filhos de joao?
prolog.consultar_e_mostrar("pai(joao, X)")

# Consulta 3: Quem são os avós de ana?
prolog.consultar_e_mostrar("avo(X, ana)")

# Consulta 4: Quem são os irmãos de maria?
prolog.consultar_e_mostrar("irmao(maria, X)")

# Consulta 5: Quem são os tios de ana?
prolog.consultar_e_mostrar("tio(X, ana)")

# Consulta 6: José é avô de quem?
prolog.consultar_e_mostrar("avo(jose, X)")

# Consulta 7: Quem são ancestrais de carlos?
prolog.consultar_e_mostrar("ancestral(X, carlos)")

# Consulta 8: Carlos é ancestral de quem? (ninguém)
prolog.consultar_e_mostrar("ancestral(carlos, X)")

# ═══════════════════════════════════════════════════════════════════
# 5. VISUALIZAÇÃO DA ÁRVORE GENEALÓGICA
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ÁRVORE GENEALÓGICA")
print("="*70)

print("\n👨‍👩‍👧‍👦 FAMÍLIA:")
print("""
        José ─┬─ Lúcia
              │
        ┌─────┴─────┐
        │           │
      João ─┬─ Clara    Paulo
            │
      ┌─────┴─────┐
      │           │
    Maria ─┬─ Pedro
           │
     ┌─────┴─────┐
     │           │
    Ana       Carlos
""")

# ═══════════════════════════════════════════════════════════════════
# 6. ESTATÍSTICAS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ESTATÍSTICAS DO SISTEMA")
print("="*70)

print(f"\n📊 BASE DE CONHECIMENTO:")
print(f"   Fatos: {len(prolog.fatos)}")
print(f"   Regras: {len(prolog.regras)}")

# Contar por predicado
predicados = defaultdict(int)
for fato in prolog.fatos:
    predicados[fato[0]] += 1

print(f"\n📈 FATOS POR PREDICADO:")
for pred, count in sorted(predicados.items()):
    print(f"   {pred}: {count}")

# ═══════════════════════════════════════════════════════════════════
# 7. COMPARAÇÃO: LÓGICA PROLOG VS IMPERATIVA
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("LÓGICA DECLARATIVA vs IMPERATIVA")
print("="*70)

print("\n🎯 PROLOG (Declarativa):")
print("   avo(X, Y) :- pai(X, Z), pai(Z, Y).")
print("   ")
print("   ✅ DECLARA o quê queremos (não como)")
print("   ✅ Unificação e backtracking automáticos")
print("   ✅ Código conciso e legível")

print("\n💻 PYTHON IMPERATIVA:")
print("""
   def encontrar_avos(neto):
       avos = []
       for pai_neto in pais[neto]:
           for avo in pais[pai_neto]:
               avos.append(avo)
       return avos
""")
print("   ✅ DESCREVE como fazer (passo a passo)")
print("   ✅ Controle explícito do fluxo")
print("   ✅ Mais verboso")

# ═══════════════════════════════════════════════════════════════════
# 8. RELATÓRIO FINAL
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("RELATÓRIO FINAL - MINI-PROLOG")
print("="*70)

print(f"\n🏗️ IMPLEMENTAÇÃO:")
print(f"   • Linguagem: Python")
print(f"   • Paradigma: Lógico (inspirado em Prolog)")
print(f"   • Técnicas: Unificação, Backtracking")
print(f"   • Suporte: Fatos, Regras, Consultas")

print(f"\n🔍 FUNCIONALIDADES:")
print(f"   ✅ Unificação de termos")
print(f"   ✅ Backtracking automático")
print(f"   ✅ Variáveis (maiúsculas)")
print(f"   ✅ Regras recursivas")
print(f"   ✅ Múltiplas soluções")

print(f"\n🎯 APLICAÇÕES REAIS:")
print(f"   • Sistemas especialistas")
print(f"   • Processamento de linguagem natural")
print(f"   • Prova automática de teoremas")
print(f"   • Planejamento (IA clássica)")
print(f"   • Bancos de dados dedutivos")

print(f"\n💡 VANTAGENS DE PROLOG:")
print(f"   ✅ DECLARATIVO: Foco no 'o quê', não no 'como'")
print(f"   ✅ CONCISO: Código muito compacto")
print(f"   ✅ BACKTRACKING: Busca automática")
print(f"   ✅ PATTERN MATCHING: Unificação poderosa")

print(f"\n⚠️ DESAFIOS:")
print(f"   ❌ Curva de aprendizado (paradigma diferente)")
print(f"   ❌ Performance (backtracking pode ser lento)")
print(f"   ❌ Debug difícil (ordem das regras importa)")

print("\n✅ MINI-PROLOG COMPLETO!")
