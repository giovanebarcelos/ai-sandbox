# GO1615-22LoraLowrankAdaptation
from peft import LoraConfig, get_peft_model


if __name__ == "__main__":
    config = LoraConfig(r=8, lora_alpha=32, target_modules=["q","v"])
    model = get_peft_model(base_model, config)
