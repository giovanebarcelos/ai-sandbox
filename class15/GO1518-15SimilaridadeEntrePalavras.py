# GO1518-15SimilaridadeEntrePalavras
from sklearn.metrics.pairwise import cosine_similarity


if __name__ == "__main__":
    v1 = model.wv['python']
    v2 = model.wv['programação']
    v3 = model.wv['banana']

    sim_12 = cosine_similarity([v1], [v2])[0][0]
    sim_13 = cosine_similarity([v1], [v3])[0][0]

    result = model.wv.most_similar(
        positive=['rei', 'mulher'],
        negative=['homem']
    )
