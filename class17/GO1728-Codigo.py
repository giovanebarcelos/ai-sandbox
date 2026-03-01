# GO1728-Codigo
prompt_few_shot_cot = """
Aqui estão alguns exemplos de raciocínio:

Q: Se uma pizza tem 8 fatias e comemos 3, quantas sobram?
A: Vamos pensar:
   1) Pizza original: 8 fatias
   2) Comidas: 3 fatias
   3) Sobram: 8 - 3 = 5 fatias
   Resposta: 5 fatias

Q: João tem 20 reais. Ele ganha mais 15 reais e gasta 12. Quanto tem agora?
A: Vamos pensar:
   1) Inicial: 20 reais
   2) Ganha: +15 reais → 20 + 15 = 35 reais
   3) Gasta: -12 reais → 35 - 12 = 23 reais
   Resposta: 23 reais

Agora resolva esta:

Q: Maria tem 3 caixas, cada uma com 7 maçãs. Ela come 4 maçãs.
Quantas maçãs sobram?
A:
"""

# LLM segue o padrão dos exemplos!
