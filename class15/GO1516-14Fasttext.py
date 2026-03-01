# GO1516-14Fasttext
from gensim.models import FastText


if __name__ == "__main__":
    model = FastText(sentences=sentencas,
                     vector_size=100,
                     window=5,
                     min_count=1)

    vetor_nova_palavra = model.wv['pythonista']
