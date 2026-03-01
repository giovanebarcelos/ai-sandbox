# GO1515-12Word2vec
  import numpy as np
  def load_glove(filepath):
      embeddings = {}
      with open(filepath, 'r', encoding='utf-8') as f:
          for line in f:
              values = line.split()
              word = values[0]
              vector = np.asarray(values[1:], dtype='float32')
              embeddings[word] = vector
