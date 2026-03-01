# GO1514-12Word2vec
from gensim.models import Word2Vec


if __name__ == "__main__":
    sentencas = [
        ['eu', 'gosto', 'de', 'python'],
        ['python', 'é', 'ótimo'],
        ['eu', 'amo', 'programação']
    ]

    model = Word2Vec(sentences=sentencas,
                     vector_size=100,
                     window=5,
                     min_count=1,
                     workers=4)

    vetor = model.wv['python']
    similares = model.wv.most_similar('python', topn=3)
