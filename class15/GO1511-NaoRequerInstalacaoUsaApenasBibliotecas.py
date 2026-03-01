# GO1511-NãoRequerInstalaçãoUsaApenasBibliotecas
  def preprocess_text(texto):
      texto = texto.lower()
      texto = texto.translate(str.maketrans('', '', string.punctuation))
      palavras = word_tokenize(texto, language='portuguese')
      stop_words = set(stopwords.words('portuguese'))
      palavras = [w for w in palavras if w not in stop_words]
      stemmer = RSLPStemmer()
      palavras = [stemmer.stem(w) for w in palavras]
      return ' '.join(palavras)
