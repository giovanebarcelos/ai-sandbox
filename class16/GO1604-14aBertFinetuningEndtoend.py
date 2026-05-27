# GO1604-14aBertFinetuningEndtoend
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

class BERTFineTuner:
    """
    Pipeline completo de fine-tuning do BERT

    Passos:
    1. Carregar BERT pré-treinado
    2. Preparar dataset (sentimento IMDB)
    3. Tokenizar e codificar
    4. Fine-tune com Trainer API
    5. Avaliar e visualizar
    """

    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        print(f"✅ Modelo {model_name} carregado")
        print(f"   Parâmetros: {sum(p.numel() for p in self.model.parameters()):,}")

    def load_data(self, dataset_name="imdb", sample_size=1000):
        """Carrega e amostra o dataset"""
        print(f"\n📥 Carregando dataset {dataset_name}...")

        dataset = load_dataset(dataset_name)

        # Amostra para treino mais rápido (demo)
        train_dataset = dataset['train'].shuffle(seed=42).select(range(sample_size))
        test_dataset = dataset['test'].shuffle(seed=42).select(range(sample_size // 5))

        print(f"   Amostras de treino: {len(train_dataset)}")
        print(f"   Amostras de teste: {len(test_dataset)}")

        return train_dataset, test_dataset

    def tokenize_function(self, examples):
        """Tokeniza textos"""
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )

    def prepare_datasets(self, train_dataset, test_dataset):
        """Tokeniza e formata os datasets"""
        print("\n🔧 Tokenizando datasets...")

        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)

        # Renomear coluna de rótulo
        train_dataset = train_dataset.rename_column('label', 'labels')
        test_dataset = test_dataset.rename_column('label', 'labels')

        # Definir formato
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

        print("   ✅ Tokenização concluída")

        return train_dataset, test_dataset

    def compute_metrics(self, pred):
        """Calcula métricas de avaliação"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)

        return {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train(self, train_dataset, test_dataset, output_dir='./bert-finetuned'):
        """Executa o fine-tune do BERT"""
        print("\n🚀 Iniciando fine-tuning...")

        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            logging_steps=50,
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
        )

        # Treinar
        train_result = trainer.train()

        print("\n✅ Treinamento concluído!")
        print(f"   Loss de treino: {train_result.training_loss:.4f}")

        # Avaliar
        eval_result = trainer.evaluate()

        print(f"\n📊 Resultados de Avaliação:")
        for key, value in eval_result.items():
            print(f"   {key}: {value:.4f}")

        return trainer, train_result, eval_result

    def predict_and_analyze(self, trainer, test_dataset):
        """Faz predições e analisa os resultados"""
        predictions = trainer.predict(test_dataset)

        preds = predictions.predictions.argmax(-1)
        labels = predictions.label_ids

        # Confusion matrix
        cm = confusion_matrix(labels, preds)

        return preds, labels, cm

# === DEMO ===

print("🎓 Demo de Fine-tuning do BERT\n")
print("="*70)

# Nota: Esta é uma demo simplificada
# Execução completa requer biblioteca datasets e GPU

print("\n📌 Visão Geral da Arquitetura:\n")
print("BERT Pré-treinado (110M parâmetros)")
print("   ↓")
print("Adicionar Cabeça de Classificação (2 classes)")
print("   ↓")
print("Fine-tune no IMDB (sentimento)")
print("   ↓")
print("Avaliar no conjunto de teste")

# Resultados simulados
print("\n📊 Resultados Simulados de Fine-tuning:\n")

epochs = [1, 2, 3]
train_loss = [0.42, 0.28, 0.18]
eval_loss = [0.35, 0.31, 0.30]
train_acc = [0.82, 0.90, 0.94]
eval_acc = [0.85, 0.88, 0.89]

for epoch, tr_loss, ev_loss, tr_acc, ev_acc in zip(epochs, train_loss, eval_loss, train_acc, eval_acc):
    print(f"Epoch {epoch}:")
    print(f"   Loss de Treino: {tr_loss:.3f} | Ac. de Treino: {tr_acc:.1%}")
    print(f"   Loss de Val.:  {ev_loss:.3f} | Ac. de Val.:  {ev_acc:.1%}")
    print()

print("\n📈 Métricas Finais:")
print("   Acurácia:   89,0%")
print("   Precisão:   88,5%")
print("   Revocação: 89,8%")
print("   F1 Score:   89,1%")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Training curves
ax = axes[0, 0]
ax.plot(epochs, train_loss, 'o-', label='Loss de Treino', linewidth=2, markersize=8, color='blue')
ax.plot(epochs, eval_loss, 's-', label='Loss de Val.', linewidth=2, markersize=8, color='red')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Loss de Treino e Validação')
ax.legend()
ax.grid(alpha=0.3)

# 2. Accuracy curves
ax = axes[0, 1]
ax.plot(epochs, train_acc, 'o-', label='Acurácia de Treino', linewidth=2, markersize=8, color='green')
ax.plot(epochs, eval_acc, 's-', label='Acurácia de Val.', linewidth=2, markersize=8, color='orange')
ax.set_xlabel('Epoch')
ax.set_ylabel('Acurácia')
ax.set_title('Acurácia de Treino e Validação')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0.7, 1.0)

# 3. Confusion matrix
ax = axes[1, 0]
cm_simulated = np.array([[85, 15], [12, 88]])
sns.heatmap(cm_simulated, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Negativo', 'Positivo'],
            yticklabels=['Negativo', 'Positivo'])
ax.set_title('Matriz de Confusão (Conjunto de Teste)')
ax.set_ylabel('Rótulo Verdadeiro')
ax.set_xlabel('Rótulo Previsto')

# 4. Learning rate schedule
ax = axes[1, 1]
steps = np.linspace(0, 1000, 100)
lr = 2e-5 * np.ones_like(steps)  # LR constante
lr_warmup = 2e-5 * np.minimum(steps / 100, 1)  # Com warmup

ax.plot(steps, lr, label='LR Constante', linewidth=2, linestyle='--', alpha=0.5)
ax.plot(steps, lr_warmup, label='LR com Warmup', linewidth=2)
ax.set_xlabel('Passos de Treino')
ax.set_ylabel('Learning Rate')
ax.set_title('Escalonamento de Learning Rate')
ax.legend()
ax.grid(alpha=0.3)
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()
plt.show()
print("\n📊 Gráfico salvo: bert_finetuning.png")

print("\n✅ Demo de fine-tuning do BERT concluída!")
print("\n💡 PRINCIPAIS APRENDIZADOS:")
print("   - Comece com o BERT pré-treinado (110M parâmetros)")
print("   - Adicione cabeça específica da tarefa (classificação, QA, NER)")
print("   - Fine-tune com learning rate pequena (2e-5)")
print("   - Poucas epochs são suficientes (2-4)")
print("   - Obtém resultados fortes com poucos dados")
print("\n💡 MELHORES PRÁTICAS:")
print("   - Use warmup no learning rate")
print("   - Gradient clipping (max_grad_norm=1.0)")
print("   - Weight decay para regularização")
print("   - Salve o melhor modelo (early stopping)")
print("   - Avalie no conjunto de teste reservado")
