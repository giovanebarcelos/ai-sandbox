# GO2103-StableDiffusionComDiffusersHuggingFace
# ═══════════════════════════════════════════════════════════════════
# STABLE DIFFUSION COM DIFFUSERS (HUGGING FACE)
# ═══════════════════════════════════════════════════════════════════

# Instalação:
# pip install diffusers transformers accelerate safetensors

from diffusers import StableDiffusionPipeline
import torch

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

# ─── Carregar modelo ───
model_id = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16  # Usa FP16 para economizar memória
)

# Mover para GPU se disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

# ─── Gerar imagem a partir de texto ───
prompt = "A beautiful sunset over mountains, oil painting style, highly detailed"

negative_prompt = "blurry, low quality, distorted, ugly"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=50,      # Mais steps = melhor qualidade
    guidance_scale=7.5,           # Quanto seguir o prompt (1-20)
    height=512,
    width=512
).images[0]

# Salvar
image.save("generated_landscape.png")
print("Imagem gerada e salva!")

# Visualizar
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title(prompt, fontsize=10, wrap=True)
plt.axis('off')
plt.tight_layout()
plt.show()

# ─── Geração em batch ───
prompts = [
    "A futuristic city at night, cyberpunk style",
    "A cute robot playing guitar, cartoon style",
    "Ancient temple in jungle, photorealistic"
]

images = pipe(prompts, num_inference_steps=30).images

for idx, img in enumerate(images):
    img.save(f"generated_{idx}.png")

# Visualizar o batch em um grid
fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
for ax, p, img in zip(axes, prompts, images):
    ax.imshow(img)
    ax.set_title(p, fontsize=9, wrap=True)
    ax.axis('off')
plt.suptitle('Geração em Batch', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ─── Image-to-Image (modificar imagem existente) ───
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to(device)

# Carregar imagem inicial
init_image = Image.open("sketch.png").convert("RGB")
init_image = init_image.resize((512, 512))

# Transformar sketch em foto realista
prompt = "photorealistic landscape, detailed, 4k"

result = img2img_pipe(
    prompt=prompt,
    image=init_image,
    strength=0.75,  # 0.0=mantém original, 1.0=completa transformação
    guidance_scale=7.5,
    num_inference_steps=50
).images[0]

result.save("sketch_to_photo.png")

# Visualizar: imagem original (sketch) vs resultado (img2img)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(init_image)
axes[0].set_title('Sketch (entrada)', fontsize=12, fontweight='bold')
axes[0].axis('off')
axes[1].imshow(result)
axes[1].set_title('Resultado (img2img)', fontsize=12, fontweight='bold')
axes[1].axis('off')
plt.suptitle(prompt, fontsize=12)
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# DICAS PARA PROMPTS:
# ✅ Seja específico: "golden retriever puppy" vs "dog"
# ✅ Estilo: "oil painting", "photorealistic", "anime"
# ✅ Qualidade: "highly detailed", "4k", "professional"
# ✅ Iluminação: "soft lighting", "dramatic shadows"
# ✅ Negative prompts: lista o que NÃO quer
# ═══════════════════════════════════════════════════════════════════
