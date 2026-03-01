# GO1502-4NormalizaçãoDeTexto
from nltk.corpus import stopwords
import string


if __name__ == "__main__":
    texto = "O Python é ÓTIMO para NLP!"
    texto = texto.lower()
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    print(texto)

    palavras = texto.split()
    stop_words = set(stopwords.words('portuguese'))
    palavras_filtradas = [w for w in palavras if w not in stop_words]
    print(palavras_filtradas)
