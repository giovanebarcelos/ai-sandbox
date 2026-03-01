# GO0422-DetectarSpamTradicional
# Abordagem tradicional - regras explícitas
def detectar_spam_tradicional(email):
    if "viagra" in email or "winner" in email:
        return "SPAM"
    if email.count("$$$") > 2:
        return "SPAM"
    return "NÃO-SPAM"
