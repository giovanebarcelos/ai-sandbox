# GO1542-NãoRequerInstalaçãoDeErro
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

class SentimentAnalyzer:
    """Análise de sentimento com múltiplos métodos"""

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()

    def analyze_vader(self, text):
        """VADER: bom para textos de redes sociais"""
        scores = self.vader.polarity_scores(text)
        return scores

    def analyze_textblob(self, text):
        """TextBlob: análise baseada em léxico"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 a 1
        subjectivity = blob.sentiment.subjectivity  # 0 a 1
        return {'polarity': polarity, 'subjectivity': subjectivity}

    def classify_sentiment(self, compound_score):
        """Classificar sentimento baseado no score"""
        if compound_score >= 0.05:
            return 'Positive'
        elif compound_score <= -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    def analyze_batch(self, texts):
        """Analisar múltiplos textos"""
        results = []

        for text in texts:
            vader_scores = self.analyze_vader(text)
            textblob_scores = self.analyze_textblob(text)

            result = {
                'text': text,
                'vader_compound': vader_scores['compound'],
                'vader_pos': vader_scores['pos'],
                'vader_neu': vader_scores['neu'],
                'vader_neg': vader_scores['neg'],
                'textblob_polarity': textblob_scores['polarity'],
                'textblob_subjectivity': textblob_scores['subjectivity'],
                'vader_sentiment': self.classify_sentiment(vader_scores['compound']),
                'textblob_sentiment': self.classify_sentiment(textblob_scores['polarity'])
            }
            results.append(result)

        return pd.DataFrame(results)

    def visualize_comparison(self, df):
        """Visualizar comparação entre métodos"""

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        # 1. VADER Compound Scores
        colors_vader = ['red' if s < -0.05 else 'gray' if s < 0.05 else 'green' 
                       for s in df['vader_compound']]
        axes[0, 0].barh(range(len(df)), df['vader_compound'], color=colors_vader, alpha=0.7)
        axes[0, 0].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[0, 0].axvline(x=-0.05, color='red', linestyle=':', alpha=0.5)
        axes[0, 0].axvline(x=0.05, color='green', linestyle=':', alpha=0.5)
        axes[0, 0].set_xlabel('VADER Compound Score', fontsize=11)
        axes[0, 0].set_ylabel('Text Index', fontsize=11)
        axes[0, 0].set_title('VADER Sentiment Scores', fontsize=13, fontweight='bold')
        axes[0, 0].grid(axis='x', alpha=0.3)

        # 2. TextBlob Polarity
        colors_tb = ['red' if s < -0.05 else 'gray' if s < 0.05 else 'green' 
                    for s in df['textblob_polarity']]
        axes[0, 1].barh(range(len(df)), df['textblob_polarity'], color=colors_tb, alpha=0.7)
        axes[0, 1].axvline(x=0, color='black', linestyle='--', linewidth=1)
        axes[0, 1].set_xlabel('TextBlob Polarity', fontsize=11)
        axes[0, 1].set_ylabel('Text Index', fontsize=11)
        axes[0, 1].set_title('TextBlob Sentiment Scores', fontsize=13, fontweight='bold')
        axes[0, 1].grid(axis='x', alpha=0.3)

        # 3. Scatter: VADER vs TextBlob
        axes[0, 2].scatter(df['vader_compound'], df['textblob_polarity'], 
                          c=range(len(df)), cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        axes[0, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 2].axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 2].set_xlabel('VADER Compound', fontsize=11)
        axes[0, 2].set_ylabel('TextBlob Polarity', fontsize=11)
        axes[0, 2].set_title('VADER vs TextBlob Correlation', fontsize=13, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)

        # Correlação
        corr = df['vader_compound'].corr(df['textblob_polarity'])
        axes[0, 2].text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                       transform=axes[0, 2].transAxes, fontsize=10,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 4. VADER Component Breakdown (stacked bar)
        vader_components = df[['vader_pos', 'vader_neu', 'vader_neg']].values
        indices = np.arange(len(df))
        axes[1, 0].barh(indices, vader_components[:, 0], label='Positive', color='green', alpha=0.7)
        axes[1, 0].barh(indices, vader_components[:, 1], left=vader_components[:, 0], 
                       label='Neutral', color='gray', alpha=0.7)
        axes[1, 0].barh(indices, vader_components[:, 2], 
                       left=vader_components[:, 0] + vader_components[:, 1],
                       label='Negative', color='red', alpha=0.7)
        axes[1, 0].set_xlabel('Proportion', fontsize=11)
        axes[1, 0].set_ylabel('Text Index', fontsize=11)
        axes[1, 0].set_title('VADER Sentiment Components', fontsize=13, fontweight='bold')
        axes[1, 0].legend(loc='upper right', fontsize=9)
        axes[1, 0].set_xlim(0, 1)

        # 5. Subjectivity vs Polarity (TextBlob)
        scatter = axes[1, 1].scatter(df['textblob_subjectivity'], df['textblob_polarity'],
                                    c=df['vader_compound'], cmap='RdYlGn', s=150, 
                                    alpha=0.7, edgecolors='black', linewidth=1.5)
        axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Subjectivity', fontsize=11)
        axes[1, 1].set_ylabel('Polarity', fontsize=11)
        axes[1, 1].set_title('Subjectivity vs Polarity (color = VADER)', fontsize=13, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 1], label='VADER Compound')

        # 6. Agreement between methods
        agreement = (df['vader_sentiment'] == df['textblob_sentiment']).sum()
        disagreement = len(df) - agreement

        axes[1, 2].pie([agreement, disagreement], 
                      labels=[f'Agree\n({agreement})', f'Disagree\n({disagreement})'],
                      colors=['#2ecc71', '#e74c3c'],
                      autopct='%1.1f%%',
                      startangle=90,
                      explode=(0.05, 0.05),
                      shadow=True)
        axes[1, 2].set_title('VADER vs TextBlob Agreement', fontsize=13, fontweight='bold')

        plt.tight_layout()
        plt.savefig('sentiment_analysis_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

# Textos de teste
texts = [
    "I absolutely love this product! It's amazing and works perfectly!",
    "This is terrible. Worst experience ever. Complete waste of money.",
    "The item arrived on time. It does what it's supposed to do.",
    "Not bad, but could be better. Some features are missing.",
    "BEST PURCHASE EVER!!! 😍 So happy with it!",
    "Disappointed. Expected more for the price.",
    "It's okay. Nothing special, nothing terrible.",
    "Fantastic quality! Highly recommend to everyone!",
    "Broke after one week. Don't buy this garbage.",
    "Good value for money. Happy with the purchase.",
]

print("="*70)
print("SENTIMENT ANALYSIS - VADER vs TEXTBLOB")
print("="*70)

# Analisar
analyzer = SentimentAnalyzer()
results_df = analyzer.analyze_batch(texts)

# Exibir resultados
print(f"\n📊 RESULTADOS DETALHADOS:\n")
for idx, row in results_df.iterrows():
    print(f"{idx+1}. \"{row['text'][:50]}...\"")
    print(f"   VADER: {row['vader_sentiment']:8s} (compound: {row['vader_compound']:6.3f})")
    print(f"   TextBlob: {row['textblob_sentiment']:8s} (polarity: {row['textblob_polarity']:6.3f})")
    if row['vader_sentiment'] != row['textblob_sentiment']:
        print(f"   ⚠️  DISAGREEMENT!")
    print()

# Visualizar
analyzer.visualize_comparison(results_df)

# Estatísticas
print("="*70)
print("📈 ESTATÍSTICAS")
print("="*70)

vader_counts = results_df['vader_sentiment'].value_counts()
textblob_counts = results_df['textblob_sentiment'].value_counts()

print(f"\nVADER Distribution:")
print(vader_counts)
print(f"\nTextBlob Distribution:")
print(textblob_counts)

agreement_rate = (results_df['vader_sentiment'] == results_df['textblob_sentiment']).mean()
print(f"\nAgreement Rate: {agreement_rate:.1%}")

print(f"\n✅ Análise completa!")
