# GO1616-23Quantização
from transformers import BitsAndBytesConfig


if __name__ == "__main__":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModel.from_pretrained(name, quantization_config=quantization_config)
