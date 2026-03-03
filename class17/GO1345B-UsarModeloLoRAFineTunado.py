# GO1345B-UsarModeloFineTunado
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Carregar modelo base


if __name__ == "__main__":
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        device_map="auto"
    )

    # Carregar adaptadores LoRA
    model = PeftModel.from_pretrained(
        base_model,
        "./lora_adapters"  # Seus adaptadores treinados
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    # Inferência
    prompt = "Explique o que é LoRA em uma frase:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=100)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
