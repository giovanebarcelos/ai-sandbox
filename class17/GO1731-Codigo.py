# GO1731-Codigo
prompt_sentiment = """
Classifique o sentimento como POSITIVO, NEGATIVO ou NEUTRO:

Texto: "Adorei este produto! Qualidade excepcional."
Sentimento: POSITIVO

Texto: "Péssimo atendimento, nunca mais volto."
Sentimento: NEGATIVO

Texto: "O produto chegou hoje."
Sentimento: NEUTRO

Texto: "Estou muito feliz com minha compra, recomendo!"
Sentimento: POSITIVO

Agora classifique:
Texto: "O produto é ok, nada de especial."
Sentimento:
"""

response = ollama.chat(model='llama3.2', messages=[{'role': 'user', 'content': prompt_sentiment}])
print(response['message']['content'])  # RESULTADO: NEUTRO
