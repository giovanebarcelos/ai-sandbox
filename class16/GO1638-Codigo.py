# GO1638-Codigo
# ═══════════════════════════════════════════════════════
# INSTALAR DEPENDÊNCIAS
# ═══════════════════════════════════════════════════════

"""
pip install transformers datasets peft bitsandbytes accelerate
"""

# ═══════════════════════════════════════════════════════
# 1. IMPORTS
# ═══════════════════════════════════════════════════════

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import load_dataset
import numpy as np

print(f"CUDA disponível: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# ═══════════════════════════════════════════════════════
# 2. CARREGAR MODELO BASE
# ═══════════════════════════════════════════════════════

model_name = "distilgpt2"  # Pequeno para demo (82M parâmetros)
# Alternativas: "gpt2" (124M), "gpt2-medium" (355M), "meta-llama/Llama-2-7b-hf" (7B)

print(f"\n🤖 Carregando modelo base: {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT não tem pad_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # FP16 para economia de memória
    device_map="auto"
)

# Ver parâmetros totais
total_params = sum(p.numel() for p in model.parameters())
print(f"   Total de parâmetros: {total_params:,} ({total_params/1e6:.1f}M)")

# ═══════════════════════════════════════════════════════
# 3. CONFIGURAR LoRA
# ═══════════════════════════════════════════════════════

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  # Language modeling
    r=8,                           # Rank (4-64, maior = mais capacidade)
    lora_alpha=32,                 # Scaling factor (tipicamente 2×r)
    lora_dropout=0.1,              # Dropout para regularização
    target_modules=[               # Quais camadas adicionar LoRA
        "c_attn",                  # Query, Key, Value projection
        "c_proj",                  # Output projection
    ],
    inference_mode=False,          # Training mode
    bias="none"                    # Não treinar bias
)

print(f"\n🔧 Configuração LoRA:")
print(f"   Rank: {lora_config.r}")
print(f"   Alpha: {lora_config.lora_alpha}")
print(f"   Target modules: {lora_config.target_modules}")

# Aplicar LoRA
model = get_peft_model(model, lora_config)

# Ver parâmetros treináveis
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
all_params = sum(p.numel() for p in model.parameters())
trainable_percent = 100 * trainable_params / all_params

print(f"\n📊 Parâmetros após LoRA:")
print(f"   Treináveis: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
print(f"   Total: {all_params:,} ({all_params/1e6:.1f}M)")
print(f"   Porcentagem treinável: {trainable_percent:.2f}%")
print(f"   Redução: {all_params/trainable_params:.1f}x menos parâmetros!")

# ═══════════════════════════════════════════════════════
# 4. PREPARAR DATASET
# ═══════════════════════════════════════════════════════

print(f"\n📚 Carregando dataset (E-commerce customer service)...")

# Dataset personalizado (exemplo)
train_texts = [
    "Customer: How can I return my product?\nAgent: You can return within 30 days with receipt.",
    "Customer: Where is my order?\nAgent: Your order is being shipped and will arrive in 2-3 days.",
    "Customer: Can I get a refund?\nAgent: Yes, refunds are processed within 5-7 business days.",
    "Customer: Is this product in stock?\nAgent: Yes, we have this item available for immediate shipping.",
    "Customer: What are your business hours?\nAgent: We're open Monday-Friday 9AM-5PM EST.",
] * 20  # Repetir para ter mais exemplos (100 total)

# Tokenizar
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        padding="max_length"
    )

from datasets import Dataset

train_dataset = Dataset.from_dict({"text": train_texts})
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
train_dataset.set_format("torch")

print(f"   Dataset size: {len(train_dataset)} exemplos")

# ═══════════════════════════════════════════════════════
# 5. CONFIGURAR TREINO
# ═══════════════════════════════════════════════════════

training_args = TrainingArguments(
    output_dir="./lora-gpt2-customer-service",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Batch efetivo = 4×4 = 16
    learning_rate=2e-4,              # LoRA usa LR maior que full fine-tune
    fp16=True,                       # Mixed precision
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    warmup_steps=50,
    optim="adamw_torch",
    report_to="none"                 # Desabilitar wandb/tensorboard
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM (não masked)
)

# ═══════════════════════════════════════════════════════
# 6. TREINAR
# ═══════════════════════════════════════════════════════

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

print(f"\n🚀 Iniciando treino LoRA...")
print(f"   Épocas: {training_args.num_train_epochs}")
print(f"   Batch size: {training_args.per_device_train_batch_size}")
print(f"   Learning rate: {training_args.learning_rate}")

trainer.train()

print(f"\n✅ Treino concluído!")

# ═══════════════════════════════════════════════════════
# 7. SALVAR ADAPTADOR LoRA
# ═══════════════════════════════════════════════════════

lora_adapter_path = "./lora-gpt2-customer-service-adapter"
model.save_pretrained(lora_adapter_path)
tokenizer.save_pretrained(lora_adapter_path)

print(f"\n💾 Adaptador LoRA salvo em: {lora_adapter_path}")

# Ver tamanho
import os
adapter_size = sum(os.path.getsize(f"{lora_adapter_path}/{f}") 
                   for f in os.listdir(lora_adapter_path) 
                   if os.path.isfile(f"{lora_adapter_path}/{f}"))
print(f"   Tamanho do adaptador: {adapter_size / 1e6:.1f} MB")
print(f"   (Modelo completo seria: ~{total_params*2/1e6:.0f} MB)")

# ═══════════════════════════════════════════════════════
# 8. TESTAR GERAÇÃO
# ═══════════════════════════════════════════════════════

def generate_response(prompt, model, tokenizer, max_length=100):
    """Gerar resposta com modelo fine-tuned"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

print(f"\n🧪 TESTANDO GERAÇÃO:")
print("="*80)

test_prompts = [
    "Customer: How can I track my order?\nAgent:",
    "Customer: Do you offer warranties?\nAgent:",
    "Customer: What payment methods do you accept?\nAgent:"
]

for prompt in test_prompts:
    print(f"\n📝 Prompt:\n{prompt}")
    response = generate_response(prompt, model, tokenizer)
    print(f"\n🤖 Resposta:\n{response}")
    print("-"*80)

# ═══════════════════════════════════════════════════════
# 9. CARREGAR ADAPTADOR EM OUTRO MOMENTO
# ═══════════════════════════════════════════════════════

print(f"\n🔄 EXEMPLO: Carregar adaptador posteriormente...")

# Carregar modelo base
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Carregar adaptador LoRA
model_with_adapter = PeftModel.from_pretrained(
    base_model,
    lora_adapter_path
)

print(f"   ✅ Modelo + adaptador carregado!")

# OPCIONAL: Merge adaptador ao modelo base
print(f"\n🔀 MERGE: Integrar adaptador permanentemente...")
merged_model = model_with_adapter.merge_and_unload()

print(f"   ✅ Adaptador integrado ao modelo!")
print(f"   Agora pode salvar como modelo completo:")
# merged_model.save_pretrained("./gpt2-customer-service-merged")
