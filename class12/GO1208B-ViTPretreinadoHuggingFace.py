# GO1208B-PretreinadosHuggingFace
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import requests

# Carregar modelo pré-treinado


if __name__ == "__main__":
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224'
    )
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        'google/vit-base-patch16-224'
    )

    # Carregar imagem
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    # Preprocessar
    inputs = feature_extractor(images=image, return_tensors="pt")

    # Predição
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()

    print(f"Predicted class: {model.config.id2label[predicted_class]}")
    # Predicted class: Egyptian cat
