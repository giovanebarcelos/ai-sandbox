# GO1523-19NamedEntityRecognitionNer
import spacy

nlp = spacy.load('pt_core_news_sm')
doc = nlp("João mora em São Paulo e trabalha na Google")

for ent in doc.ents:
    print(f'{ent.text}: {ent.label_}')
