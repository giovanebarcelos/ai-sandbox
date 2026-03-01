# GO1604-14aBertFinetuningEndtoend
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns

class BERTFineTuner:
    """
    Complete BERT fine-tuning pipeline

    Steps:
    1. Load pre-trained BERT
    2. Prepare dataset (IMDB sentiment)
    3. Tokenize and encode
    4. Fine-tune with Trainer API
    5. Evaluate and visualize
    """

    def __init__(self, model_name="bert-base-uncased", num_labels=2):
        self.model_name = model_name
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        print(f"✅ Loaded {model_name}")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def load_data(self, dataset_name="imdb", sample_size=1000):
        """Load and sample dataset"""
        print(f"\n📥 Loading {dataset_name} dataset...")

        dataset = load_dataset(dataset_name)

        # Sample for faster training (demo)
        train_dataset = dataset['train'].shuffle(seed=42).select(range(sample_size))
        test_dataset = dataset['test'].shuffle(seed=42).select(range(sample_size // 5))

        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")

        return train_dataset, test_dataset

    def tokenize_function(self, examples):
        """Tokenize texts"""
        return self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=512
        )

    def prepare_datasets(self, train_dataset, test_dataset):
        """Tokenize and format datasets"""
        print("\n🔧 Tokenizing datasets...")

        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)

        # Rename label column
        train_dataset = train_dataset.rename_column('label', 'labels')
        test_dataset = test_dataset.rename_column('label', 'labels')

        # Set format
        train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

        print("   ✅ Tokenization complete")

        return train_dataset, test_dataset

    def compute_metrics(self, pred):
        """Compute evaluation metrics"""
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
        """Fine-tune BERT"""
        print("\n🚀 Starting fine-tuning...")

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

        # Train
        train_result = trainer.train()

        print("\n✅ Training complete!")
        print(f"   Training loss: {train_result.training_loss:.4f}")

        # Evaluate
        eval_result = trainer.evaluate()

        print(f"\n📊 Evaluation Results:")
        for key, value in eval_result.items():
            print(f"   {key}: {value:.4f}")

        return trainer, train_result, eval_result

    def predict_and_analyze(self, trainer, test_dataset):
        """Make predictions and analyze results"""
        predictions = trainer.predict(test_dataset)

        preds = predictions.predictions.argmax(-1)
        labels = predictions.label_ids

        # Confusion matrix
        cm = confusion_matrix(labels, preds)

        return preds, labels, cm

# === DEMO ===

print("🎓 BERT Fine-tuning Demo\n")
print("="*70)

# Note: This is a simplified demo
# Full execution requires datasets library and GPU

print("\n📌 Architecture Overview:\n")
print("Pre-trained BERT (110M params)")
print("   ↓")
print("Add Classification Head (2 classes)")
print("   ↓")
print("Fine-tune on IMDB (sentiment)")
print("   ↓")
print("Evaluate on test set")

# Simulated results
print("\n📊 Simulated Fine-tuning Results:\n")

epochs = [1, 2, 3]
train_loss = [0.42, 0.28, 0.18]
eval_loss = [0.35, 0.31, 0.30]
train_acc = [0.82, 0.90, 0.94]
eval_acc = [0.85, 0.88, 0.89]

for epoch, tr_loss, ev_loss, tr_acc, ev_acc in zip(epochs, train_loss, eval_loss, train_acc, eval_acc):
    print(f"Epoch {epoch}:")
    print(f"   Train Loss: {tr_loss:.3f} | Train Acc: {tr_acc:.1%}")
    print(f"   Eval Loss:  {ev_loss:.3f} | Eval Acc:  {ev_acc:.1%}")
    print()

print("\n📈 Final Metrics:")
print("   Accuracy:  89.0%")
print("   Precision: 88.5%")
print("   Recall:    89.8%")
print("   F1 Score:  89.1%")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Training curves
ax = axes[0, 0]
ax.plot(epochs, train_loss, 'o-', label='Train Loss', linewidth=2, markersize=8, color='blue')
ax.plot(epochs, eval_loss, 's-', label='Eval Loss', linewidth=2, markersize=8, color='red')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training & Validation Loss')
ax.legend()
ax.grid(alpha=0.3)

# 2. Accuracy curves
ax = axes[0, 1]
ax.plot(epochs, train_acc, 'o-', label='Train Accuracy', linewidth=2, markersize=8, color='green')
ax.plot(epochs, eval_acc, 's-', label='Eval Accuracy', linewidth=2, markersize=8, color='orange')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title('Training & Validation Accuracy')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0.7, 1.0)

# 3. Confusion matrix
ax = axes[1, 0]
cm_simulated = np.array([[85, 15], [12, 88]])
sns.heatmap(cm_simulated, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'])
ax.set_title('Confusion Matrix (Test Set)')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

# 4. Learning rate schedule
ax = axes[1, 1]
steps = np.linspace(0, 1000, 100)
lr = 2e-5 * np.ones_like(steps)  # Constant LR
lr_warmup = 2e-5 * np.minimum(steps / 100, 1)  # With warmup

ax.plot(steps, lr, label='Constant LR', linewidth=2, linestyle='--', alpha=0.5)
ax.plot(steps, lr_warmup, label='LR with Warmup', linewidth=2)
ax.set_xlabel('Training Steps')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule')
ax.legend()
ax.grid(alpha=0.3)
ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

plt.tight_layout()
plt.savefig('bert_finetuning.png', dpi=150, bbox_inches='tight')
print("\n📊 Gráfico salvo: bert_finetuning.png")

print("\n✅ BERT fine-tuning demo completo!")
print("\n💡 KEY TAKEAWAYS:")
print("   - Start with pre-trained BERT (110M params)")
print("   - Add task-specific head (classification, QA, NER)")
print("   - Fine-tune with small learning rate (2e-5)")
print("   - Few epochs needed (2-4)")
print("   - Achieves strong results with little data")
print("\n💡 BEST PRACTICES:")
print("   - Use learning rate warmup")
print("   - Gradient clipping (max_grad_norm=1.0)")
print("   - Weight decay for regularization")
print("   - Save best model (early stopping)")
print("   - Evaluate on held-out test set")
