# GO1536-36Prática15aDatasetImdbCompleto
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from sklearn.model_selection import train_test_split

# Baixar recursos do NLTK


if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    # Carregar dataset IMDB
    # Opção 1: Usando Keras
    from tensorflow.keras.datasets import imdb

    # Carregar como texto (não como índices)
    # Vamos usar uma abordagem alternativa para ter texto completo
    print("Carregando dataset IMDB...")

    # Opção 2: Carregar de arquivo CSV ou usar dataset do Hugging Face
    from datasets import load_dataset

    # Carregar IMDB do Hugging Face
    dataset = load_dataset('imdb')
    train_data = dataset['train']
    test_data = dataset['test']

    # Converter para DataFrame
    df_train = pd.DataFrame({
        'text': train_data['text'],
        'sentiment': train_data['label']
    })

    df_test = pd.DataFrame({
        'text': test_data['text'],
        'sentiment': test_data['label']
    })

    print(f"Training samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")
    print("\nClass distribution:")
    print(df_train['sentiment'].value_counts())

    # Visualizar exemplos
    print("\n" + "="*80)
    print("EXEMPLO DE REVIEW POSITIVO:")
    print("="*80)
    positive_example = df_train[df_train['sentiment'] == 1].iloc[0]
    print(f"Sentimento: {positive_example['sentiment']}")
    print(f"Texto: {positive_example['text'][:300]}...")

    print("\n" + "="*80)
    print("EXEMPLO DE REVIEW NEGATIVO:")
    print("="*80)
    negative_example = df_train[df_train['sentiment'] == 0].iloc[0]
    print(f"Sentimento: {negative_example['sentiment']}")
    print(f"Texto: {negative_example['text'][:300]}...")

    # Função de preprocessamento completo
    def preprocess_text(text, remove_stopwords=True, lemmatize=True):
        """
        Preprocessa texto com limpeza, tokenização, remoção de stopwords e lematização
        """
        # 1. Converter para minúsculas
        text = text.lower()

        # 2. Remover HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # 3. Remover URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # 4. Remover menções e hashtags (se for Twitter)
        text = re.sub(r'@\w+|#\w+', '', text)

        # 5. Remover números
        text = re.sub(r'\d+', '', text)

        # 6. Remover caracteres especiais e pontuação (manter espaços)
        text = re.sub(r'[^\w\s]', '', text)

        # 7. Remover espaços múltiplos
        text = re.sub(r'\s+', ' ', text).strip()

        # 8. Tokenização
        tokens = word_tokenize(text)

        # 9. Remover stopwords (opcional)
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]

        # 10. Lematização (opcional)
        if lemmatize:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]

        # 11. Remover tokens muito curtos (< 2 caracteres)
        tokens = [word for word in tokens if len(word) > 2]

        return ' '.join(tokens)

    # Aplicar preprocessamento
    print("\n" + "="*80)
    print("APLICANDO PREPROCESSAMENTO...")
    print("="*80)

    df_train['text_clean'] = df_train['text'].apply(preprocess_text)
    df_test['text_clean'] = df_test['text'].apply(preprocess_text)

    # Comparar antes e depois
    print("\nANTES DO PREPROCESSAMENTO:")
    print(df_train['text'].iloc[0][:200])
    print("\nDEPOIS DO PREPROCESSAMENTO:")
    print(df_train['text_clean'].iloc[0][:200])

    # Estatísticas de preprocessamento
    df_train['word_count_original'] = df_train['text'].apply(lambda x: len(x.split()))
    df_train['word_count_clean'] = df_train['text_clean'].apply(lambda x: len(x.split()))

    print("\n" + "="*80)
    print("ESTATÍSTICAS DO PREPROCESSAMENTO:")
    print("="*80)
    print(f"Palavras médias (original): {df_train['word_count_original'].mean():.2f}")
    print(f"Palavras médias (limpo): {df_train['word_count_clean'].mean():.2f}")
    print(f"Redução: {(1 - df_train['word_count_clean'].mean() / df_train['word_count_original'].mean()) * 100:.1f}%")

    # Salvar dados preprocessados
    df_train.to_csv('imdb_train_preprocessed.csv', index=False)
    df_test.to_csv('imdb_test_preprocessed.csv', index=False)

    print("\n✅ Dataset IMDB carregado e preprocessado com sucesso!")
    print(f"📁 Arquivos salvos: imdb_train_preprocessed.csv e imdb_test_preprocessed.csv")
