# GO1629-36cAvançadoComparaçãoBertVsGpt
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import matplotlib.pyplot as plt

class TransformerComparison:
    """Comparar BERT (Encoder) vs GPT (Decoder)"""

    def __init__(self):
        print("🔄 Carregando modelos...")

        # BERT (Encoder-only)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        # GPT-2 (Decoder-only)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

        self.bert_model.eval()
        self.gpt2_model.eval()

        print("✅ Modelos carregados!")

    def compare_attention_masks(self, text="The cat sat on the mat"):
        """Comparar máscaras de atenção bidirectional vs causal"""

        # BERT: Attention bidirectional (vê tudo)
        bert_tokens = self.bert_tokenizer(text, return_tensors='pt')
        bert_input_ids = bert_tokens['input_ids']
        seq_len = bert_input_ids.shape[1]

        # GPT-2: Causal mask (só vê passado)
        gpt2_tokens = self.gpt2_tokenizer(text, return_tensors='pt')
        gpt2_input_ids = gpt2_tokens['input_ids']

        # Criar máscaras
        # BERT: matriz de 1s (todos veem todos)
        bert_mask = torch.ones((seq_len, seq_len))

        # GPT: triangular inferior (só vê tokens anteriores)
        gpt_mask = torch.tril(torch.ones((seq_len, seq_len)))

        # Visualizar
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # BERT
        im1 = axes[0].imshow(bert_mask, cmap='Greens', aspect='auto')
        axes[0].set_title('BERT: Bidirectional Attention\n(Todos tokens veem todos tokens)', fontsize=12)
        axes[0].set_xlabel('Key Position')
        axes[0].set_ylabel('Query Position')
        axes[0].set_xticks(range(seq_len))
        axes[0].set_yticks(range(seq_len))
        tokens_bert = self.bert_tokenizer.convert_ids_to_tokens(bert_input_ids[0])
        axes[0].set_xticklabels(tokens_bert, rotation=45, ha='right')
        axes[0].set_yticklabels(tokens_bert)
        plt.colorbar(im1, ax=axes[0], label='Can Attend (1=Yes)')

        # GPT
        im2 = axes[1].imshow(gpt_mask, cmap='Blues', aspect='auto')
        axes[1].set_title('GPT: Causal (Autoregressive) Attention\n(Tokens só veem passado)', fontsize=12)
        axes[1].set_xlabel('Key Position')
        axes[1].set_ylabel('Query Position')
        axes[1].set_xticks(range(seq_len))
        axes[1].set_yticks(range(seq_len))
        tokens_gpt = self.gpt2_tokenizer.convert_ids_to_tokens(gpt2_input_ids[0])
        axes[1].set_xticklabels(tokens_gpt, rotation=45, ha='right')
        axes[1].set_yticklabels(tokens_gpt)
        plt.colorbar(im2, ax=axes[1], label='Can Attend (1=Yes)')

        plt.tight_layout()
        plt.savefig('bert_vs_gpt_attention_masks.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n📊 DIFERENÇAS DE ARQUITETURA:")
        print(f"\n🔵 BERT (Encoder):")
        print(f"   - Bidirectional: vê contexto completo")
        print(f"   - Melhor para: classificação, NER, Q&A")
        print(f"   - Tokens visíveis para 'cat': {bert_mask[2].sum().item()}/{seq_len}")

        print(f"\n🔴 GPT (Decoder):")
        print(f"   - Causal: só vê tokens anteriores")
        print(f"   - Melhor para: geração de texto")
        print(f"   - Tokens visíveis para 'cat': {gpt_mask[2].sum().item()}/{seq_len}")

    def compare_embeddings(self, sentences):
        """Comparar embeddings gerados"""

        fig, axes = plt.subplots(len(sentences), 2, figsize=(14, 4*len(sentences)))

        for idx, sentence in enumerate(sentences):
            # BERT embeddings
            with torch.no_grad():
                bert_tokens = self.bert_tokenizer(sentence, return_tensors='pt')
                bert_outputs = self.bert_model(**bert_tokens)
                bert_last_hidden = bert_outputs.last_hidden_state[0].numpy()

            # GPT-2 embeddings
            with torch.no_grad():
                gpt2_tokens = self.gpt2_tokenizer(sentence, return_tensors='pt')
                gpt2_outputs = self.gpt2_model.transformer(**gpt2_tokens)
                gpt2_last_hidden = gpt2_outputs.last_hidden_state[0].numpy()

            # Plot BERT
            im1 = axes[idx, 0].imshow(bert_last_hidden.T, cmap='viridis', aspect='auto')
            axes[idx, 0].set_title(f'BERT Embeddings\n"{sentence}"')
            axes[idx, 0].set_ylabel('Hidden Dimension')
            axes[idx, 0].set_xlabel('Token Position')
            tokens_bert = self.bert_tokenizer.tokenize(sentence)
            axes[idx, 0].set_xticks(range(1, len(tokens_bert)+1))
            axes[idx, 0].set_xticklabels(tokens_bert, rotation=45, ha='right')
            plt.colorbar(im1, ax=axes[idx, 0])

            # Plot GPT-2
            im2 = axes[idx, 1].imshow(gpt2_last_hidden.T, cmap='plasma', aspect='auto')
            axes[idx, 1].set_title(f'GPT-2 Embeddings\n"{sentence}"')
            axes[idx, 1].set_ylabel('Hidden Dimension')
            axes[idx, 1].set_xlabel('Token Position')
            tokens_gpt = self.gpt2_tokenizer.tokenize(sentence)
            axes[idx, 1].set_xticks(range(len(tokens_gpt)))
            axes[idx, 1].set_xticklabels(tokens_gpt, rotation=45, ha='right')
            plt.colorbar(im2, ax=axes[idx, 1])

        plt.tight_layout()
        plt.savefig('bert_vs_gpt_embeddings.png', dpi=300, bbox_inches='tight')
        plt.show()

    def benchmark_tasks(self):
        """Comparar adequação para diferentes tarefas"""
        tasks = {
            'Sentiment\nAnalysis': {'BERT': 95, 'GPT': 70},
            'Named Entity\nRecognition': {'BERT': 92, 'GPT': 65},
            'Question\nAnswering': {'BERT': 90, 'GPT': 75},
            'Text\nGeneration': {'BERT': 40, 'GPT': 95},
            'Text\nCompletion': {'BERT': 35, 'GPT': 98},
            'Summarization': {'BERT': 55, 'GPT': 88}
        }

        task_names = list(tasks.keys())
        bert_scores = [tasks[t]['BERT'] for t in task_names]
        gpt_scores = [tasks[t]['GPT'] for t in task_names]

        x = np.arange(len(task_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, bert_scores, width, label='BERT (Encoder)', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, gpt_scores, width, label='GPT (Decoder)', color='#e74c3c', alpha=0.8)

        ax.set_ylabel('Performance Score', fontsize=12)
        ax.set_title('BERT vs GPT: Adequação para Diferentes Tarefas\n(Score relativo - não oficial)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(task_names)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=0.3)

        # Adicionar valores nas barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('bert_vs_gpt_tasks.png', dpi=300, bbox_inches='tight')
        plt.show()

# Executar comparações


if __name__ == "__main__":
    comparator = TransformerComparison()

    # 1. Máscaras de atenção
    print("\n" + "="*60)
    print("1️⃣  MÁSCARAS DE ATENÇÃO")
    print("="*60)
    comparator.compare_attention_masks("The quick brown fox jumps")

    # 2. Embeddings
    print("\n" + "="*60)
    print("2️⃣  COMPARAÇÃO DE EMBEDDINGS")
    print("="*60)
    sentences = [
        "I love machine learning",
        "The weather is beautiful today"
    ]
    comparator.compare_embeddings(sentences)

    # 3. Benchmark de tarefas
    print("\n" + "="*60)
    print("3️⃣  ADEQUAÇÃO PARA TAREFAS")
    print("="*60)
    comparator.benchmark_tasks()

    print("\n✅ Análise completa!")
    print("\n📊 RESUMO:")
    print("   BERT (Encoder): Entendimento contextual bidirectional")
    print("   GPT (Decoder): Geração sequencial autoregressive")
    print("   Use BERT para: classificação, extração de informação")
    print("   Use GPT para: geração de texto, completação")
