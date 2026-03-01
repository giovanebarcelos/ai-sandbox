# GO1607-17HuggingFaceTransformers
from transformers import pipeline, AutoModel, AutoTokenizer


if __name__ == "__main__":
    classifier = pipeline("sentiment-analysis")
    result = classifier("I love this product!")

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
