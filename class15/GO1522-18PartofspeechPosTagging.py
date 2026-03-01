# GO1522-18PartofspeechPosTagging
import spacy

nlp = spacy.load('pt_core_news_sm')
doc = nlp("O gato subiu no telhado")

for token in doc:
    print(f'{token.text}: {token.pos_}')
