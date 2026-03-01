# GO1730-Codigo
prompt_few_shot = """
Extraia Nome, Cargo e Empresa do texto:

Texto: "João Silva é engenheiro na Microsoft"
Nome: João Silva
Cargo: Engenheiro
Empresa: Microsoft

Texto: "Maria Santos trabalha como designer na Apple"
Nome: Maria Santos
Cargo: Designer
Empresa: Apple

Texto: "Carlos Pereira atua como gerente de projetos no Google"
Nome: Carlos Pereira
Cargo: Gerente de Projetos
Empresa: Google

Agora extraia:
Texto: "Ana Costa é cientista de dados na Amazon"
"""

response = ollama.chat(
    model='llama3.2',
    messages=[{'role': 'user', 'content': prompt_few_shot}]
)

print(response['message']['content'])

# RESULTADO:
# Nome: Ana Costa
# Cargo: Cientista de Dados
# Empresa: Amazon
