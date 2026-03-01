# GO1530-25QuestionAnswering
  from transformers import pipeline
  qa = pipeline('question-answering', model='deepset/bert-base-...')
  result = qa(question="Quem inventou Python?",
              context="Python foi criado por Guido van Rossum...")
  print(result['answer'])
