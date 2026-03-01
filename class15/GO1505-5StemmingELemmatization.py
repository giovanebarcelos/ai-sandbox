# GO1505-5StemmingELemmatization
import spacy


if __name__ == "__main__":
    nlp = spacy.load('pt_core_news_sm')
    doc = nlp("correndo")
    print(doc[0].lemma_)
