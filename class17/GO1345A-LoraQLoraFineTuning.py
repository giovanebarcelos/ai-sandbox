# GO1345A-34aLoraEQloraFineTuning
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch


if __name__ == "__main__":
    print("🔧 FINE-TUNING COM QLoRA")
    print("=" * 70)

    # ──────────────────────────────────────────
    # 1. CONFIGURAR QUANTIZAÇÃO (4-bit)
    # ──────────────────────────────────────────

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                      # Quantização 4-bit
        bnb_4bit_quant_type="nf4",              # NormalFloat 4-bit
        bnb_4bit_compute_dtype=torch.float16,   # Compute em FP16
        bnb_4bit_use_double_quant=True          # Double quantization
    )

    print("✅ Quantização 4-bit configurada (NF4)")

    # ──────────────────────────────────────────
    # 2. CARREGAR MODELO BASE (Llama 2 7B)
    # ──────────────────────────────────────────

    model_id = "meta-llama/Llama-2-7b-hf"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,  # Aplica quantização
        device_map="auto",               # Distribui entre GPUs
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"✅ Modelo carregado: {model_id}")
    print(f"   Memória GPU: ~4.5 GB (vs ~28 GB sem quantização)")

    # ──────────────────────────────────────────
    # 3. PREPARAR MODELO PARA K-BIT TRAINING
    # ──────────────────────────────────────────

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    print("✅ Modelo preparado para treinamento QLoRA")

    # ──────────────────────────────────────────
    # 4. CONFIGURAR LoRA
    # ──────────────────────────────────────────

    lora_config = LoraConfig(
        r=16,                           # Rank (8, 16, 32)
        lora_alpha=32,                  # Scaling factor (2*r típico)
        target_modules=[                # Aplicar LoRA nestas camadas
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,              # Dropout para regularização
        bias="none",                    # Não treinar biases
        task_type="CAUSAL_LM"           # Tarefa: language modeling
    )

    print(f"✅ LoRA configurado (r={lora_config.r}, alpha={lora_config.lora_alpha})")

    # ──────────────────────────────────────────
    # 5. APLICAR LoRA AO MODELO
    # ──────────────────────────────────────────

    model = get_peft_model(model, lora_config)

    # Verificar parâmetros treináveis
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\n📊 Parâmetros:")
    print(f"   Total: {total_params:,}")
    print(f"   Treináveis: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"   Redução: {total_params/trainable_params:.0f}x menos parâmetros!\n")

    # ──────────────────────────────────────────
    # 6. TREINAR (exemplo simplificado)
    # ──────────────────────────────────────────

    from transformers import TrainingArguments, Trainer

    # Dados de exemplo (substituir por seu dataset)
    train_dataset = ...  # Seu dataset tokenizado

    training_args = TrainingArguments(
        output_dir="./llama2-qlora-finetuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # Batch efetivo: 16
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit"  # Optimizer otimizado para QLoRA
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    # trainer.train()

    print("✅ Treinamento configurado!")
    print("\n💡 Para treinar: trainer.train()")
    print("💾 Modelo final ocupa ~50 MB (só adaptadores LoRA)")

    # ──────────────────────────────────────────
    # 7. SALVAR ADAPTADORES LoRA
    # ──────────────────────────────────────────

    # model.save_pretrained("./lora_adapters")
    # Salva APENAS os pesos LoRA (~50 MB)
    # Modelo base permanece inalterado

    print("\n📦 Salvar adaptadores: model.save_pretrained('./lora_adapters')")
