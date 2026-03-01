# GO1527-23TopicModeling
  from gensim import corpora, models
  dictionary = corpora.Dictionary(documentos_tokenizados)
  corpus = [dictionary.doc2bow(doc) for doc in docs]
  lda = models.LdaModel(corpus, num_topics=10, id2word=dictionary)
  topics = lda.print_topics()
