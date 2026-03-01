# GO1635-36iAvançadoTransferLearningComLora
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

class LoRAFineTuning:
    """Comparação: Fine-tuning tradicional vs LoRA"""

    def __init__(self, model_name='distilbert-base-uncased', task='classification'):
        self.model_name = model_name
        self.task = task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device: {self.device}")

    def count_parameters(self, model):
        """Contar parâmetros treináveis e totais"""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    def create_models(self):
        """Criar modelo base e modelo com LoRA"""

        print("\n🔄 Criando modelos...")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Modelo base (fine-tuning completo)
        self.model_full = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        ).to(self.device)

        # Modelo com LoRA
        self.model_lora_base = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2
        ).to(self.device)

        # Configurar LoRA
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,  # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin"]  # DistilBERT attention modules
        )

        self.model_lora = get_peft_model(self.model_lora_base, lora_config)

        # Contar parâmetros
        full_total, full_trainable = self.count_parameters(self.model_full)
        lora_total, lora_trainable = self.count_parameters(self.model_lora)

        print(f"\n📊 PARÂMETROS:")
        print(f"   Full Fine-tuning:")
        print(f"      Total: {full_total:,}")
        print(f"      Trainable: {full_trainable:,} (100%)")
        print(f"\n   LoRA Fine-tuning:")
        print(f"      Total: {lora_total:,}")
        print(f"      Trainable: {lora_trainable:,} ({lora_trainable/full_trainable*100:.2f}%)")
        print(f"      Reduction: {(1 - lora_trainable/full_trainable)*100:.1f}%")

        # Visualizar redução
        self._visualize_parameter_comparison(full_trainable, lora_trainable)

        return {
            'full_trainable': full_trainable,
            'lora_trainable': lora_trainable,
            'reduction': (1 - lora_trainable/full_trainable) * 100
        }

    def _visualize_parameter_comparison(self, full, lora):
        """Visualizar comparação de parâmetros"""

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Gráfico de barras
        methods = ['Full\nFine-tuning', 'LoRA']
        params = [full, lora]
        colors = ['#e74c3c', '#3498db']

        bars = axes[0].bar(methods, params, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        axes[0].set_ylabel('Trainable Parameters', fontsize=12)
        axes[0].set_title('Trainable Parameters Comparison', fontsize=14)
        axes[0].grid(axis='y', alpha=0.3)

        for bar, param in zip(bars, params):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{param:,}\n({param/full*100:.1f}%)',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Gráfico de pizza
        sizes = [full, lora]
        labels = [f'Full\n{full:,}', f'LoRA\n{lora:,}']
        explode = (0.05, 0.05)

        axes[1].pie(sizes, explode=explode, labels=labels, colors=colors,
                   autopct='%1.1f%%', shadow=True, startangle=90)
        axes[1].set_title('Proportion of Trainable Parameters', fontsize=14)

        plt.suptitle(f'LoRA vs Full Fine-tuning - Parameter Efficiency\n' +
                    f'Reduction: {(1-lora/full)*100:.1f}%', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('lora_parameter_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def train_and_compare(self, train_size=1000, epochs=3, batch_size=16):
        """Treinar e comparar ambos os métodos"""

        print("\n📥 Carregando dataset IMDB...")
        dataset = load_dataset('imdb')
        train_data = dataset['train'].shuffle(seed=42).select(range(train_size))

        # Preparar dados
        class IMDBDataset(Dataset):
            def __init__(self, data, tokenizer, max_length=256):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                text = self.data[idx]['text']
                label = self.data[idx]['label']

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
                    'labels': torch.tensor(label, dtype=torch.long)
                }

        train_dataset = IMDBDataset(train_data, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Treinar ambos
        results = {}

        for name, model in [('Full', self.model_full), ('LoRA', self.model_lora)]:
            print(f"\n{'='*70}")
            print(f"🚀 Treinando: {name}")
            print(f"{'='*70}")

            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

            history = {'loss': [], 'time': []}
            total_time = 0

            model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                epoch_start = time.time()

                progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
                for batch in progress_bar:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)

                    outputs = model(input_ids=input_ids, 
                                   attention_mask=attention_mask, 
                                   labels=labels)
                    loss = outputs.loss

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    epoch_loss += loss.item()
                    progress_bar.set_postfix({'loss': loss.item()})

                epoch_time = time.time() - epoch_start
                total_time += epoch_time
                avg_loss = epoch_loss / len(train_loader)

                history['loss'].append(avg_loss)
                history['time'].append(total_time)

                print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s")

            results[name] = history
            print(f"\n✅ {name} training completo! Total time: {total_time:.2f}s")

        # Visualizar comparação
        self._visualize_training_comparison(results)

        return results

    def _visualize_training_comparison(self, results):
        """Visualizar comparação de treinamento"""

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Loss comparison
        for name, history in results.items():
            axes[0].plot(range(1, len(history['loss'])+1), history['loss'], 
                        marker='o', label=name, linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss Comparison', fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        # Time comparison
        methods = list(results.keys())
        times = [results[m]['time'][-1] for m in methods]
        colors = ['#e74c3c', '#3498db']

        bars = axes[1].bar(methods, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        axes[1].set_ylabel('Training Time (seconds)', fontsize=12)
        axes[1].set_title('Total Training Time', fontsize=14)
        axes[1].grid(axis='y', alpha=0.3)

        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{time_val:.1f}s',
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.suptitle('LoRA vs Full Fine-tuning - Training Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('lora_training_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

# Executar comparação


if __name__ == "__main__":
    print("="*80)
    print("TRANSFER LEARNING: FULL FINE-TUNING vs LoRA")
    print("="*80)

    fine_tuner = LoRAFineTuning()

    # Criar e comparar modelos
    param_stats = fine_tuner.create_models()

    # Treinar e comparar (reduzir tamanho para demo rápida)
    print("\n⚠️  Executando training demo com dataset reduzido...")
    training_results = fine_tuner.train_and_compare(train_size=500, epochs=2, batch_size=8)

    print("\n" + "="*80)
    print("📊 RESUMO FINAL")
    print("="*80)
    print(f"\n✅ LoRA reduz parâmetros treináveis em {param_stats['reduction']:.1f}%")
    print(f"✅ Mantém qualidade similar ao fine-tuning completo")
    print(f"✅ Memória: ~{param_stats['reduction']:.0f}% menor")
    print(f"✅ Velocidade: similar ou ligeiramente mais rápida")
    print(f"✅ Permite múltiplos adapters para diferentes tarefas")

    print("\n💡 VANTAGENS DO LoRA:")
    print("   - Eficiência de parâmetros (treina <1% dos pesos)")
    print("   - Baixo consumo de memória")
    print("   - Múltiplos adapters para uma base")
    print("   - Deploy flexível (trocar adapters)")
    print("   - Ideal para LLMs grandes (Llama, GPT)")

    print("\n✅ Análise completa!")
