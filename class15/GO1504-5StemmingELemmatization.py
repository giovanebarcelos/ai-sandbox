# GO1504-5StemmingELemmatization
from nltk.stem import PorterStemmer, RSLPStemmer


if __name__ == "__main__":
    stemmer_pt = RSLPStemmer()
    print(stemmer_pt.stem("correndo"))
