# GO0423-Sklearn
# Abordagem ML - aprende padrões automaticamente
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Dados de exemplo: emails e seus rótulos (0 = não spam, 1 = spam)
emails = [
    "Ganhe dinheiro fácil agora mesmo! Clique aqui!",
    "Você ganhou um prêmio de R$10.000! Resgate já!",
    "Oferta imperdível: produtos com 90% de desconto!",
    "Parabéns! Você foi selecionado para ganhar um iPhone!",
    "URGENTE: Sua conta será bloqueada, clique no link!",
    "Oi, como foi sua reunião de ontem?",
    "Segue em anexo o relatório do projeto.",
    "Podemos marcar uma call amanhã às 15h?",
    "Obrigado pelo feedback, vou ajustar o documento.",
    "Lembrete: reunião de equipe na sexta-feira.",
    "Grátis! Ganhe um carro! Sem sorteio!",
    "Clique agora e ganhe créditos grátis!",
    "Você foi sorteado! Reclame seu prêmio!",
    "Seu pedido foi enviado. Código de rastreio: ABC123",
    "Por favor, revise o contrato anexo.",
    "Feliz aniversário! Abraços da equipe.",
    "Confirme sua presença no evento de quinta.",
    "Lucre R$5000 por dia trabalhando em casa!",
    "Oferta exclusiva só para você! 80% OFF!",
    "Atenção! Sua fatura vence amanhã.",
]

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0]

if __name__ == "__main__":
    # 1. Converter textos em features numéricas (bag of words)
    vetorizador = CountVectorizer()
    X = vetorizador.fit_transform(emails)
    
    # 2. Dividir em treino e teste
    emails_treino, emails_teste, labels_treino, labels_teste = train_test_split(
        X, labels, test_size=0.3, random_state=42
    )
    
    # 3. Treinar o modelo com exemplos
    modelo = MultinomialNB()
    modelo.fit(emails_treino, labels_treino)
    
    # 4. Avaliar o modelo
    predicoes = modelo.predict(emails_teste)
    print(f"Acurácia: {accuracy_score(labels_teste, predicoes):.2%}")
    print("\nRelatório de Classificação:")
    print(classification_report(labels_teste, predicoes, target_names=["Não Spam", "Spam"]))
    
    # 5. Usar o modelo treinado com novos emails
    novos_emails = [
        "Reunião confirmada para amanhã às 10h",
        "GANHE DINHEIRO RÁPIDO! CLIQUE AQUI AGORA!",
        "Segue o documento que você pediu",
    ]
    novos_emails_vetorizados = vetorizador.transform(novos_emails)
    predicao = modelo.predict(novos_emails_vetorizados)
    
    print("\n--- Predições para novos emails ---")
    for email, pred in zip(novos_emails, predicao):
        resultado = "SPAM" if pred == 1 else "Não Spam"
        print(f"'{email[:50]}...' → {resultado}")
