# GO1639-Transformers
# ═══════════════════════════════════════════════════════
# QLoRA: LoRA + 4-bit Quantization
# ═══════════════════════════════════════════════════════

from transformers import BitsAndBytesConfig

# Configuração de quantização 4-bit


if __name__ == "__main__":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                      # Carregar em 4-bit
        bnb_4bit_quant_type="nf4",              # NormalFloat4 (melhor que INT4)
        bnb_4bit_compute_dtype=torch.float16,  # Compute em FP16
        bnb_4bit_use_double_quant=True          # Quantização dupla (economia extra)
    )

    # Carregar modelo em 4-bit
    model_4bit = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",  # Agora possível em GPU 8GB!
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Aplicar LoRA (mesma config)
    model_4bit = get_peft_model(model_4bit, lora_config)

    # Treinar normalmente
    # trainer = Trainer(model=model_4bit, ...)
    # trainer.train()

    print("✅ Llama-7B treinando em GPU 8GB com QLoRA!")
