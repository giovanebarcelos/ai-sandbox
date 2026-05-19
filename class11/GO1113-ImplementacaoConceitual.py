# GO1113-ImplementacaoConceitual
# Conjuntos Fuzzy Tipo-2 — extensão que modela incerteza nas próprias
# funções de pertinência (a MF tem uma "faixa" em vez de uma curva fixa).
# Exemplo: TEMPERATURA 'quente' com MF inferior (certeza mínima) e
# superior (certeza máxima), refletindo que especialistas divergem.
#
# ATENÇÃO: este código é CONCEITUAL. A biblioteca 'type2fuzzy' pode não
# estar disponível (pip install type2fuzzy). Para uso prático, prefira
# scikit-fuzzy (GO1108/GO1110) que implementa Tipo-1 de forma robusta.
#
# Para outro problema: ajuste lower_mf e upper_mf para refletir a
# incerteza do seu domínio (ex: diagnóstico médico com opiniões divergentes).
# pip install type2fuzzy
from type2fuzzy import Type2FuzzySet

# Criar conjunto Type-2


if __name__ == "__main__":
    temp_quente = Type2FuzzySet(
        lower_mf=triangular([20, 25, 30]),
        upper_mf=triangular([18, 25, 32])
    )
