# GO2103-StableDiffusionComDiffusersHuggingFace
# ═══════════════════════════════════════════════════════════════════
# STABLE DIFFUSION COM DIFFUSERS (HUGGING FACE)
# ═══════════════════════════════════════════════════════════════════

# Instalação:
# pip install diffusers transformers accelerate safetensors

from diffusers import StableDiffusionPipeline
import torch

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

# ─── Geração em batch ───
prompts = [
    "A futuristic city at night, cyberpunk style",
    "A cute robot playing guitar, cartoon style",
    "Ancient temple in jungle, photorealistic"
]

images = pipe(prompts, num_inference_steps=30).images

for idx, img in enumerate(images):
    img.save(f"generated_{idx}.png")

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

# ═══════════════════════════════════════════════════════════════════
# DICAS PARA PROMPTS:
# ✅ Seja específico: "golden retriever puppy" vs "dog"
# ✅ Estilo: "oil painting", "photorealistic", "anime"
# ✅ Qualidade: "highly detailed", "4k", "professional"
# ✅ Iluminação: "soft lighting", "dramatic shadows"
# ✅ Negative prompts: lista o que NÃO quer
# ═══════════════════════════════════════════════════════════════════
