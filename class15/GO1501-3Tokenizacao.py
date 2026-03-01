# GO1501-3Tokenização
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize


if __name__ == "__main__":
    texto = "Olá mundo! Como vai você?"
    palavras = word_tokenize(texto, language='portuguese')
    print(palavras)

    sentencas = sent_tokenize(texto, language='portuguese')
    print(sentencas)
