# GO1532-TraducaoNeuralMarianMT
from transformers import MarianMTModel, MarianTokenizer


if __name__ == "__main__":
    model_name = 'Helsinki-NLP/opus-mt-en-pt'
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    text = "Hello, how are you?"
    translated = model.generate(**tokenizer(text, return_tensors="pt"))
    print(tokenizer.decode(translated[0], skip_special_tokens=True))
