# GO1208B-PretreinadosHuggingFace
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import requests
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass

# Carregar modelo pré-treinado


if __name__ == "__main__":
    # from_pretrained: baixa pesos do HuggingFace Hub — modelo já treinado em 1.2M imagens ImageNet
    # 'google/vit-base-patch16-224': ViT-Base com patches 16×16 para imagens 224×224
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224'
    )
    # ViTFeatureExtractor: pré-processa a imagem (redimensiona, normaliza) para o formato que o ViT espera
    feature_extractor = ViTFeatureExtractor.from_pretrained(
        'google/vit-base-patch16-224'
    )

    # Carregar imagem
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    # Preprocessar: converte PIL Image em tensor PyTorch normalizado para o ViT
    inputs = feature_extractor(images=image, return_tensors="pt")

    # torch.no_grad(): desliga o cálculo de gradientes — economiza memória na inferência
    with torch.no_grad():
        outputs = model(**inputs)
    # logits: pontuações brutas antes do softmax (uma por classe ImageNet)
    logits = outputs.logits
    # argmax(-1): classe com maior pontuação — índice da predição mais provável
    predicted_class = logits.argmax(-1).item()

    print(f"Predicted class: {model.config.id2label[predicted_class]}")
    # Predicted class: Egyptian cat

    # ─── VISUALIZAÇÃO: IMAGEM + TOP-5 PREDIÇÕES ───
    # softmax: transforma logits em probabilidades (soma = 1) para todas as 1000 classes
    probabilities = torch.softmax(logits, dim=-1)[0]
    # topk: retorna os 5 maiores valores e seus índices — top-5 predições do modelo
    top5_probs, top5_indices = torch.topk(probabilities, 5)
    # id2label: mapeia índice numérico de classe para rótulo legível (ex: 285 → 'Egyptian cat')
    top5_labels = [model.config.id2label[idx.item()] for idx in top5_indices]
    top5_values = top5_probs.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Imagem original
    axes[0].imshow(image)
    axes[0].set_title(f'Imagem de Entrada\nPredição: {model.config.id2label[predicted_class]}',
                      fontsize=12)
    axes[0].axis('off')

    # Top-5 predições
    colors = ['#e15759' if i == 0 else '#4e79a7' for i in range(5)]
    bars = axes[1].barh(range(5), top5_values * 100, color=colors, edgecolor='black')
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels([lbl[:30] for lbl in top5_labels], fontsize=10)
    axes[1].set_xlabel('Probabilidade (%)')
    axes[1].set_title('Top-5 Predições - ViT-Base/16 (ImageNet)', fontsize=12)
    axes[1].invert_yaxis()
    for bar, val in zip(bars, top5_values):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'{val*100:.2f}%', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    # ─── VISUALIZAÇÃO: PATCHES DO ViT ───
    img_array = np.array(image.resize((224, 224)))  # redimensionar para tamanho padrão do ViT
    patch_size = 16  # cada patch tem 16×16 pixels
    num_patches = 224 // patch_size  # 14 patches por lado → 14×14 = 196 patches total

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_array)
    for i in range(0, 224 + 1, patch_size):
        axes[0].axhline(i - 0.5, color='yellow', linewidth=0.4, alpha=0.7)
        axes[0].axvline(i - 0.5, color='yellow', linewidth=0.4, alpha=0.7)
    axes[0].set_title(f'Imagem dividida em {num_patches}×{num_patches} = {num_patches**2} patches', fontsize=11)
    axes[0].axis('off')

    # Mostrar alguns patches individuais
    patches_grid = []
    for r in range(0, 4):
        for c in range(0, 4):
            patch = img_array[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size]
            patches_grid.append(patch)

    axes[1].axis('off')
    axes[1].set_title('Primeiros 16 patches (4×4 da região superior)', fontsize=11)
    inner_grid = axes[1].get_position()
    for idx, patch in enumerate(patches_grid):
        ax_inset = fig.add_axes([inner_grid.x0 + (idx % 4) * inner_grid.width / 4,
                                  inner_grid.y0 + (3 - idx // 4) * inner_grid.height / 4,
                                  inner_grid.width / 4 - 0.005,
                                  inner_grid.height / 4 - 0.01])
        ax_inset.imshow(patch)
        ax_inset.axis('off')

    plt.suptitle('Vision Transformer (ViT) - Processamento de Imagem', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
