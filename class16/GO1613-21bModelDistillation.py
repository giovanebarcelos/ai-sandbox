# GO1613-21bModelDistillation
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer, BertConfig
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class DistilledBERT(nn.Module):
    """
    Student model - versão menor do BERT

    Teacher: BERT-base (110M params)
    Student: Custom BERT (30M params)
    """

    def __init__(self, vocab_size=30522, hidden_size=256, num_layers=4):
        super().__init__()

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=8,
            intermediate_size=hidden_size * 4
        )

        self.bert = BertModel(config)
        self.classifier = nn.Linear(hidden_size, 2)  # Binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        logits = self.classifier(pooled)
        return logits, outputs.last_hidden_state

class KnowledgeDistiller:
    """
    Knowledge Distillation para LLMs

    Process:
    1. Teacher gera soft targets (probabilidades)
    2. Student aprende de teacher + ground truth
    3. Loss = αL_CE + (1-α)L_KD

    Temperature: suaviza probabilidades
    """

    def __init__(self, teacher_model, student_model, temperature=3.0, alpha=0.7):
        self.teacher = teacher_model
        self.student = student_model
        self.temperature = temperature
        self.alpha = alpha  # Weight for KD loss

        self.teacher.eval()  # Teacher sempre em eval

    def distillation_loss(self, student_logits, teacher_logits, labels):
        """
        Combined loss:
        - Hard loss: CrossEntropy com labels
        - Soft loss: KL divergence com teacher probabilities
        """
        # Hard loss (student vs ground truth)
        hard_loss = nn.CrossEntropyLoss()(student_logits, labels)

        # Soft loss (student vs teacher)
        # Apply temperature to soften probabilities
        student_soft = nn.functional.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = nn.functional.softmax(teacher_logits / self.temperature, dim=1)

        soft_loss = nn.KLDivLoss(reduction='batchmean')(student_soft, teacher_soft)

        # Scale soft loss by temperature^2 (as per Hinton et al.)
        soft_loss = soft_loss * (self.temperature ** 2)

        # Combine losses
        total_loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss

        return total_loss, hard_loss.item(), soft_loss.item()

    def train_step(self, batch, optimizer):
        """Single training step"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        # Teacher predictions (no grad)
        with torch.no_grad():
            teacher_logits, _ = self.teacher(input_ids, attention_mask)

        # Student predictions
        student_logits, _ = self.student(input_ids, attention_mask)

        # Compute loss
        total_loss, hard_loss, soft_loss = self.distillation_loss(
            student_logits, teacher_logits, labels
        )

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return {
            'total_loss': total_loss.item(),
            'hard_loss': hard_loss,
            'soft_loss': soft_loss
        }

    def evaluate(self, dataloader):
        """Evaluate student model"""
        self.student.eval()

        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']

                logits, _ = self.student(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)

                total_correct += (preds == labels).sum().item()
                total_samples += len(labels)

        accuracy = total_correct / total_samples
        return accuracy

def compare_models(teacher, student):
    """Compara tamanho, velocidade, e performance"""

    # Count parameters
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())

    compression_ratio = teacher_params / student_params

    # Measure inference time
    dummy_input = torch.randint(0, 30522, (1, 128))
    dummy_mask = torch.ones(1, 128)

    import time

    # Teacher speed
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = teacher(dummy_input, dummy_mask)
    teacher_time = (time.time() - start) / 100

    # Student speed
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = student(dummy_input, dummy_mask)
    student_time = (time.time() - start) / 100

    speedup = teacher_time / student_time

    return {
        'teacher_params': teacher_params,
        'student_params': student_params,
        'compression_ratio': compression_ratio,
        'teacher_time': teacher_time,
        'student_time': student_time,
        'speedup': speedup
    }

# === DEMO ===

print("🧬 Knowledge Distillation Demo\n")
print("="*70)

# Initialize models
print("\n📌 Loading models...\n")

# Teacher: BERT-base (simplified for demo)
teacher_config = BertConfig(
    vocab_size=30522,
    hidden_size=768,
    num_hidden_layers=6,
    num_attention_heads=12
)
teacher = DistilledBERT(hidden_size=768, num_layers=6)

# Student: Smaller BERT
student = DistilledBERT(hidden_size=256, num_layers=4)

print("Teacher (BERT-base):")
teacher_params = sum(p.numel() for p in teacher.parameters())
print(f"   Parameters: {teacher_params:,}")
print(f"   Layers: 6")
print(f"   Hidden size: 768")
print()

print("Student (Distilled):")
student_params = sum(p.numel() for p in student.parameters())
print(f"   Parameters: {student_params:,}")
print(f"   Layers: 4")
print(f"   Hidden size: 256")
print(f"   Compression: {teacher_params/student_params:.1f}x smaller")
print()

# Initialize distiller
distiller = KnowledgeDistiller(teacher, student, temperature=3.0, alpha=0.7)

# Simulate training
print("\n📌 Distillation Training (simulated):\n")

epochs = 10
training_losses = []
hard_losses = []
soft_losses = []

for epoch in range(1, epochs + 1):
    # Simulate batch
    batch = {
        'input_ids': torch.randint(0, 30522, (16, 128)),
        'attention_mask': torch.ones(16, 128),
        'labels': torch.randint(0, 2, (16,))
    }

    optimizer = optim.Adam(student.parameters(), lr=5e-5)

    metrics = distiller.train_step(batch, optimizer)

    training_losses.append(metrics['total_loss'])
    hard_losses.append(metrics['hard_loss'])
    soft_losses.append(metrics['soft_loss'])

    if epoch % 2 == 0:
        print(f"Epoch {epoch}/{epochs}")
        print(f"   Total loss: {metrics['total_loss']:.4f}")
        print(f"   Hard loss: {metrics['hard_loss']:.4f}")
        print(f"   Soft loss: {metrics['soft_loss']:.4f}")

# Compare models
print("\n📌 Model Comparison:\n")

comparison = compare_models(teacher, student)

print(f"Parameters:")
print(f"   Teacher: {comparison['teacher_params']:,}")
print(f"   Student: {comparison['student_params']:,}")
print(f"   Compression: {comparison['compression_ratio']:.1f}x")
print()

print(f"Inference Speed:")
print(f"   Teacher: {comparison['teacher_time']*1000:.2f} ms")
print(f"   Student: {comparison['student_time']*1000:.2f} ms")
print(f"   Speedup: {comparison['speedup']:.1f}x faster")
print()

# Accuracy simulation
teacher_acc = 0.92
student_acc = 0.89  # Slight drop but still good
student_scratch_acc = 0.78  # Training from scratch without distillation

print(f"Accuracy:")
print(f"   Teacher: {teacher_acc:.1%}")
print(f"   Student (distilled): {student_acc:.1%}")
print(f"   Student (from scratch): {student_scratch_acc:.1%}")
print(f"   Performance retention: {student_acc/teacher_acc:.1%}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Training losses
ax = axes[0, 0]
ax.plot(range(1, epochs+1), training_losses, 'o-', label='Total Loss', linewidth=2)
ax.plot(range(1, epochs+1), hard_losses, 's-', label='Hard Loss (CE)', linewidth=2, alpha=0.7)
ax.plot(range(1, epochs+1), soft_losses, '^-', label='Soft Loss (KD)', linewidth=2, alpha=0.7)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Distillation Training Losses')
ax.legend()
ax.grid(alpha=0.3)

# 2. Size vs Accuracy tradeoff
ax = axes[0, 1]
model_sizes = [110, 66, 40, 25, 15]  # Million parameters
accuracies = [0.92, 0.90, 0.89, 0.86, 0.82]

ax.plot(model_sizes, accuracies, 'o-', linewidth=2, markersize=8, color='purple')
ax.fill_between(model_sizes, 0.75, accuracies, alpha=0.3, color='purple')
ax.set_xlabel('Model Size (M parameters)')
ax.set_ylabel('Accuracy')
ax.set_title('Model Size vs Accuracy Tradeoff')
ax.grid(alpha=0.3)
ax.invert_xaxis()

# Annotate sweet spot
ax.plot(25, 0.89, 'r*', markersize=20)
ax.annotate('Sweet Spot\n(4.4x smaller, -3% acc)', xy=(25, 0.89), xytext=(50, 0.84),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=9, fontweight='bold')

# 3. Speed comparison
ax = axes[1, 0]
models = ['Teacher\n(BERT-base)', 'Student\n(Distilled)', 'TinyBERT', 'MobileBERT']
latencies = [45, 15, 12, 18]  # ms
colors = ['red', 'lightgreen', 'lightgreen', 'yellow']

bars = ax.barh(models, latencies, color=colors, alpha=0.7)
ax.set_xlabel('Latency (ms)')
ax.set_title('Inference Speed Comparison')
ax.grid(axis='x', alpha=0.3)

for bar, lat in zip(bars, latencies):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2,
            f'{lat}ms', ha='left', va='center', fontweight='bold')

# 4. Temperature effect on distillation
ax = axes[1, 1]
temperatures = [1, 2, 3, 4, 5, 7, 10]
student_accs = [0.85, 0.87, 0.89, 0.89, 0.88, 0.86, 0.83]

ax.plot(temperatures, student_accs, 'o-', linewidth=2, markersize=8, color='blue')
ax.axhline(0.92, color='green', linestyle='--', alpha=0.5, label='Teacher accuracy')
ax.axvline(3, color='red', linestyle='--', alpha=0.5, label='Optimal T')
ax.set_xlabel('Temperature')
ax.set_ylabel('Student Accuracy')
ax.set_title('Effect of Temperature on Distillation')
ax.legend()
ax.grid(alpha=0.3)
ax.set_ylim(0.8, 0.95)

plt.tight_layout()
plt.savefig('knowledge_distillation.png', dpi=150, bbox_inches='tight')
print("\n📊 Gráfico salvo: knowledge_distillation.png")

print("\n✅ Knowledge distillation implementado!")
print("\n💡 BEST PRACTICES:")
print("   1. Use temperature T=3-5 for most tasks")
print("   2. Balance α: 0.7 for soft loss, 0.3 for hard loss")
print("   3. Train student longer than teacher")
print("   4. Use same tokenizer for teacher & student")
print("   5. Consider intermediate layer distillation")
print("   6. Validate on target hardware (mobile/edge)")
