"""
GO1345B - Usar Modelo com Adaptadores LoRA Fine-Tunados
=======================================================
Demonstra como carregar um modelo base e aplicar adaptadores LoRA
treinados previamente (gerados pelo GO1345A).

Conceito:
  Após o fine-tuning com LoRA, você tem:
  - Modelo base: Llama-2 7B (~14 GB em FP16, ou ~4.5 GB em 4-bit)
  - Adaptadores: ./lora_adapters (~50 MB)

  Para inferência, carrega-se o modelo base + adapters.
  Os dois arquivos juntos funcionam como o modelo fine-tunado.

Vantagem:
  - Pode ter múltiplos adapters para diferentes tarefas
  - Base permanece inalterado: troque o adapter = troque a tarefa
  - Economiza armazenamento: 50 MB vs 14 GB por fine-tuning

Instalação (requer GPU CUDA):
  pip install transformers peft accelerate torch

Uso: python GO1345B-UsarModeloLoRAFineTunado.py
"""

import sys


def demonstrar_inferencia_lora() -> None:
    """
    Demonstra o fluxo de inferência com LoRA sem executar o modelo real.
    Mostra o código que seria executado em produção.
    """
    print("\nFLUXO DE INFERENCIA COM ADAPTADORES LORA:")
    print("─" * 60)
    print()
    print("  # 1. Carregar modelo base (congelado)")
    print("  base = AutoModelForCausalLM.from_pretrained(")
    print("      'meta-llama/Llama-2-7b-hf',")
    print("      device_map='auto'")
    print("  )")
    print()
    print("  # 2. Aplicar adaptadores LoRA treinados")
    print("  model = PeftModel.from_pretrained(")
    print("      base,")
    print("      './lora_adapters'  # Gerado pelo GO1345A")
    print("  )")
    print()
    print("  # 3. Inferência: igual a qualquer modelo Hugging Face")
    print("  inputs = tokenizer('Explique LoRA:', return_tensors='pt').to('cuda')")
    print("  outputs = model.generate(**inputs, max_new_tokens=100)")
    print("  print(tokenizer.decode(outputs[0], skip_special_tokens=True))")
    print()
    print("  # 4. Mesclar adapters (opcional) — para deploy mais rápido")
    print("  model_merged = model.merge_and_unload()")
    print("  model_merged.save_pretrained('./llama2-qlora-merged')")


def demonstrar_multiplos_adapters() -> None:
    """
    Demonstra o poder de usar múltiplos adapters LoRA com o mesmo modelo base.
    """
    print("\nVANTAGEM: MULTIPLOS ADAPTERS, UM SO MODELO BASE:")
    print("─" * 60)
    print()
    adapters = [
        ("medicina",      "Ajuda diagnóstico médico (treino em PubMed)"),
        ("juridico",      "Analisa contratos (treino em corpus jurídico)"),
        ("codigo",        "Code completion (treino em GitHub)"),
        ("financeiro",    "Análise de mercado (treino em relatórios)"),
    ]
    for nome, descricao in adapters:
        print(f"  ./adapters/{nome}/  → {descricao}")
    print()
    print("  Base: Llama-2 7B (~14 GB)  — carregado UMA vez")
    print("  Cada adapter: ~50 MB      — troca dinâmica!")
    print()
    print("  # Trocar de tarefa em tempo de execução:")
    print("  model.set_adapter('medicina')")
    print("  resposta_med = model.generate(prompt_medico)")
    print("  model.set_adapter('juridico')")
    print("  resposta_jur = model.generate(prompt_juridico)")


def executar_inferencia_real(prompt: str = "Explique o que é LoRA em uma frase:") -> None:
    """
    Executa inferência real com modelo base + adaptadores LoRA.
    Requer: GPU CUDA, transformers, peft, e o arquivo ./lora_adapters
    """
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    if not os.path.exists("./lora_adapters"):
        print("AVISO: ./lora_adapters nao encontrado.")
        print("Execute GO1345A primeiro para gerar os adaptadores.")
        return

    print(f"\nCarregando modelo base...")
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        device_map="auto"
    )

    print("Aplicando adaptadores LoRA...")
    model = PeftModel.from_pretrained(base_model, "./lora_adapters")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    print(f"\nPrompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Resposta: {resposta}")


if __name__ == "__main__":
    print("=" * 60)
    print("GO1345B - INFERENCIA COM ADAPTADORES LORA")
    print("=" * 60)

    demonstrar_inferencia_lora()
    demonstrar_multiplos_adapters()

    print()
    print("─" * 60)
    print("PARA EXECUTAR COM MODELO REAL:")
    print("─" * 60)
    print("  Requisitos:")
    print("    - GPU NVIDIA com CUDA")
    print("    - pip install transformers peft accelerate torch")
    print("    - ./lora_adapters/ gerado pelo GO1345A")
    print("    - Token Hugging Face para Llama-2 (modelo gated)")

    # Tentar executar real
    try:
        import torch
        if not torch.cuda.is_available():
            print("\nGPU CUDA nao detectada — demonstracao conceitual acima.")
        else:
            print(f"\nGPU detectada: {torch.cuda.get_device_name(0)}")
            executar_inferencia_real()
    except ImportError:
        print("\ntorch nao instalado. Execute: pip install torch")
