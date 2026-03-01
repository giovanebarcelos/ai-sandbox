# GO1630-36dAvançadoFinetuningBertParaClassif
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuração
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Usando device: {device}")

# Carregar dados AG News (4 categorias de notícias)
print("📥 Carregando AG News dataset...")
dataset = load_dataset('ag_news')

# Mapear labels
label_names = ['World', 'Sports', 'Business', 'Sci/Tech']
print(f"Classes: {label_names}")

# Amostragem para treinamento rápido
train_size = 5000
test_size = 1000
train_data = dataset['train'].shuffle(seed=42).select(range(train_size))
test_data = dataset['test'].shuffle(seed=42).select(range(test_size))

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Criar datasets
train_dataset = NewsDataset(train_data['text'], train_data['label'], tokenizer)
test_dataset = NewsDataset(test_data['text'], test_data['label'], tokenizer)

# DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Modelo
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=4
).to(device)

# Otimizador e scheduler
epochs = 3
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
total_steps = len(train_loader) * epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# Training loop
def train_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    losses = []
    correct_predictions = 0

    progress_bar = tqdm(data_loader, desc='Training')
    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        progress_bar.set_postfix({'loss': np.mean(losses), 'acc': correct_predictions.double() / len(data_loader.dataset)})

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

# Evaluation
def eval_model(model, data_loader, device):
    model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct_predictions.double() / len(data_loader.dataset)
    return accuracy, np.mean(losses), np.array(all_preds), np.array(all_labels)

# Treinar
print("\n🚀 Iniciando fine-tuning...")
history = {'train_acc': [], 'train_loss': [], 'val_acc': [], 'val_loss': []}

for epoch in range(epochs):
    print(f'\n📅 Epoch {epoch + 1}/{epochs}')

    train_acc, train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_acc, val_loss, _, _ = eval_model(model, test_loader, device)

    history['train_acc'].append(train_acc.item())
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc.item())
    history['val_loss'].append(val_loss)

    print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

# Avaliação final
print("\n📊 Avaliação Final...")
final_acc, final_loss, predictions, true_labels = eval_model(model, test_loader, device)

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names,
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - BERT Fine-tuned\nAccuracy: {final_acc:.4f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('bert_finetuning_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Classification Report
print("\n📋 Classification Report:")
print(classification_report(true_labels, predictions, target_names=label_names))

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
axes[0].plot(history['val_loss'], label='Val Loss', marker='s')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training and Validation Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
axes[1].plot(history['val_acc'], label='Val Accuracy', marker='s')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Training and Validation Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bert_finetuning_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Testar com exemplos novos
def predict(text, model, tokenizer, device):
    model.eval()
    encoding = tokenizer(
        text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    return label_names[pred_class], probs[0].cpu().numpy()

# Exemplos de teste
test_texts = [
    "The stock market crashed today with major losses across all sectors.",
    "Scientists discover new planet in distant solar system.",
    "The championship game will be played next Sunday at the stadium.",
    "UN holds emergency meeting to discuss global climate crisis."
]

print("\n🧪 Testando com exemplos novos:\n")
for text in test_texts:
    pred_label, probs = predict(text, model, tokenizer, device)
    print(f"Text: {text[:60]}...")
    print(f"Predicted: {pred_label}")
    print(f"Probabilities: {dict(zip(label_names, probs))}\n")

print("✅ Fine-tuning completo!")
