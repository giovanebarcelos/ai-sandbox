"""
GO1345A - Fine-Tuning com LoRA e QLoRA
=======================================
Demonstra a configuração de fine-tuning eficiente com LoRA/QLoRA.
Requer GPU com suporte a CUDA para executar o treinamento real.

Conceito:
  LoRA (Low-Rank Adaptation): em vez de treinar todos os ~7 bilhões de
  parâmetros do Llama-2, treina apenas ~50 MB de adaptadores de baixo rank.
  Isso reduz o custo de fine-tuning em ~100x.

  QLoRA adiciona quantização 4-bit: o modelo base fica em 4-bit (4.5 GB de
  RAM em vez de 28 GB), e os adaptadores LoRA treinam em FP16.

Instalação (requer GPU CUDA):
  pip install transformers peft bitsandbytes accelerate datasets

Uso: python GO1345A-LoraQLoraFineTuning.py
     (sem GPU, apenas exibe a configuração e sai elegantemente)
"""

import sys


def demonstrar_configuracao_qlora() -> None:
    """
    Exibe a configuração QLoRA sem necessidade de GPU.
    Mostra o que cada parâmetro significa.
    """
    print("\nCONFIGURACAO QLORA (Quantizacao 4-bit + LoRA):")
    print("─" * 60)
    print()
    print("  BitsAndBytesConfig:")
    print("    load_in_4bit=True          → Quantiza para 4-bit (NF4)")
    print("    bnb_4bit_quant_type='nf4'  → NormalFloat4 (melhor para LLMs)")
    print("    bnb_4bit_compute_dtype=fp16→ Computa em FP16 para velocidade")
    print("    bnb_4bit_use_double_quant  → Double quantization: -0.4 bpw extra")
    print()
    print("  LoraConfig:")
    print("    r=16                       → Rank: tamanho do adaptador")
    print("    lora_alpha=32              → Escaling (2×r é padrão)")
    print("    target_modules=[q_proj, k_proj, v_proj, ...]")
    print("    lora_dropout=0.05          → Regularização")
    print("    task_type='CAUSAL_LM'      → Tarefa de linguagem")
    print()
    print("  Economia de memória:")
    print("    Llama-2 7B FP32: ~28 GB VRAM")
    print("    Llama-2 7B QLoRA: ~4.5 GB VRAM  (6× menor!)")
    print("    Parâmetros LoRA: ~50 MB  (vs 14 GB modelo completo)")


def demonstrar_algebra_lora() -> None:
    """
    Demonstra a matemática do LoRA sem necessidade de GPU/PyTorch.
    W = W_0 + BA onde B ∈ R^{d×r} e A ∈ R^{r×k}, r << min(d,k).
    """
    import numpy as np

    print("\nMATEMATICA DO LORA:")
    print("─" * 60)
    print()
    print("  Camada original: W_0 ∈ R^{d×k}")
    print("  LoRA adiciona:   ΔW = B × A")
    print("  onde B ∈ R^{d×r} e A ∈ R^{r×k}, r << min(d,k)")
    print()
    print("  Forward: h = W_0·x + (B·A)·x × (α/r)")
    print()

    # Exemplo numérico com d=4, k=4, r=2
    np.random.seed(42)
    d, k, r = 4, 4, 2
    W0 = np.random.randn(d, k) * 0.1  # Pesos pré-treinados (congelados)
    A = np.random.randn(r, k) * 0.02  # Inicializado aleatório
    B = np.zeros((d, r))               # Inicializado em ZERO (importante!)
    alpha = 32
    x = np.random.randn(k)

    h_original = W0 @ x
    delta_W = B @ A
    h_lora = (W0 + delta_W * alpha / r) @ x

    print(f"  Exemplo (d={d}, k={k}, r={r}):")
    print(f"    W_0 shape: {W0.shape}  (congelado)")
    print(f"    A shape:   {A.shape}   (treinável)")
    print(f"    B shape:   {B.shape}   (treinável, init=0)")
    print(f"    Parâmetros treináveis: {A.size + B.size} de {W0.size} total")
    print(f"    Redução: {W0.size / (A.size + B.size):.1f}× menos parâmetros")
    print(f"    h_original = {h_original.round(3)}")
    print(f"    h_lora     = {h_lora.round(3)}")
    print("    (idênticos no início: B=0, então ΔW=0)")


def executar_qlora_real() -> None:
    """
    Executa o fine-tuning QLoRA real com Llama-2 (requer GPU + Hugging Face token).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import TrainingArguments, Trainer
    import torch
    import datasets

    print("\nEXECUTANDO QLORA REAL...")
    print("─" * 60)

    # 1. Configuração de quantização 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    print("Quantizacao 4-bit configurada (NF4)")

    # 2. Carregar modelo base (requer acesso ao Hugging Face Hub)
    model_id = "meta-llama/Llama-2-7b-hf"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Modelo carregado: {model_id}")

    # 3. Preparar para k-bit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # 4. Configurar LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(lora_config)

    # 5. Exibir parâmetros
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parâmetros: {total:,} total, {trainable:,} treináveis "
          f"({trainable/total*100:.2f}%)")

    # 6. Dataset de exemplo
    # Em produção: substitua por seu dataset real tokenizado
    raw_data = [
        {"text": "O que é IA? IA é a simulação de inteligência em máquinas."},
        {"text": "O que é LoRA? LoRA é uma técnica de fine-tuning eficiente."},
    ]
    train_dataset = datasets.Dataset.from_list(raw_data)
    train_dataset = train_dataset.map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=128),
        batched=True
    )

    # 7. Configuração de treinamento
    training_args = TrainingArguments(
        output_dir="./llama2-qlora-finetuned",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        optim="paged_adamw_32bit"
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)

    # Iniciar treinamento
    trainer.train()
    model.save_pretrained("./lora_adapters")
    print("Adaptadores LoRA salvos em ./lora_adapters (~50 MB)")


if __name__ == "__main__":
    print("=" * 60)
    print("GO1345A - FINE-TUNING COM LORA E QLORA")
    print("=" * 60)

    # Sempre executar a demonstração conceitual (não requer GPU)
    demonstrar_configuracao_qlora()
    demonstrar_algebra_lora()

    print()
    print("─" * 60)
    print("PARA EXECUTAR O FINE-TUNING REAL:")
    print("─" * 60)
    print("  Requisitos:")
    print("    - GPU NVIDIA com CUDA (ex: A100, RTX 3090)")
    print("    - pip install transformers peft bitsandbytes accelerate datasets")
    print("    - Token de acesso ao Hugging Face (modelo Llama-2 é gated)")
    print()
    print("  Fluxo:")
    print("    1. Carregar Llama-2 7B em 4-bit (~4.5 GB VRAM)")
    print("    2. Aplicar LoRA (r=16): apenas ~50 MB de parâmetros treináveis")
    print("    3. Treinar com seu dataset customizado")
    print("    4. Salvar apenas os adaptadores LoRA (~50 MB)")
    print("    5. Em produção: carregar base + adapters para inferência")

    # Tentar executar real se GPU disponível
    try:
        import torch
        if not torch.cuda.is_available():
            print("\nGPU CUDA nao detectada — demonstracao conceitual acima.")
            print("Para treinar, execute em ambiente com GPU.")
        else:
            print(f"\nGPU detectada: {torch.cuda.get_device_name(0)}")
            print("Tentando executar QLoRA real...")
            executar_qlora_real()
    except ImportError:
        print("\ntorch nao instalado. Execute: pip install torch")
