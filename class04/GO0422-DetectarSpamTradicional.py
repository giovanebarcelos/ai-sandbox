# GO0422-DetectarSpamTradicional
# Abordagem tradicional - regras explícitas
def detectar_spam_tradicional(email):
    if "viagra" in email or "winner" in email:
        return "SPAM"
    if email.count("$$$") > 2:
        return "SPAM"
    return "NÃO-SPAM"


if __name__ == '__main__':
    print("=== Detector de Spam por Regras ===")
    emails = [
        "Você ganhou! Compre viagra agora mesmo!",
        "Reunião amanhã às 14h na sala de conferências.",
        "PROMOÇÃO $$$ IMPERDÍVEL $$$ CLIQUE AGORA $$$",
        "Olá, segue o relatório do projeto conforme solicitado.",
        "Congratulations! You are the winner of our lottery!",
    ]
    for email in emails:
        resultado = detectar_spam_tradicional(email)
        icone = "🚫" if resultado == "SPAM" else "✅"
        print(f"  {icone} [{resultado}] {email[:50]}")
