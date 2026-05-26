# GO1117-Png
# DIAGRAMA CONCEITUAL — não é código executável.
# Ilustra como a fuzzificação de uma feature (idade=25 anos) gera 3 graus
# que substituem o valor original ao entrar em um modelo de ML.
#
# Idade 25 → [Jovem: 0.8, Adulto: 0.2, Idoso: 0.0]
# Essas 3 features substituem o "25" original, enriquecendo a representação.
#
# Para implementar em Python, veja GO1118 (código executável equivalente).
# Idade = 25 anos
# ↓ Fuzzificação
# [Jovem: 0.8, Adulto: 0.2, Idoso: 0.0]
# ↓ ML
# Usa essas 3 features ao invés de 1
