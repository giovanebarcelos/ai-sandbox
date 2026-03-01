# GO1512-NãoRequerInstalaçãoUsaApenasBibliotecas
import re


if __name__ == "__main__":
    texto = "Contato: joao@email.com ou (11) 98765-4321"
    emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', texto)
    print(emails)
