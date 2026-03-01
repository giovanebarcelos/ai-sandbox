# GO2105-StableDiffusionGeraçãoAvançadaDeImagens
# ═══════════════════════════════════════════════════════════════════
# STABLE DIFFUSION - GERAÇÃO AVANÇADA DE IMAGENS
# ═══════════════════════════════════════════════════════════════════

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image
import numpy as np

# ─── 1. CONFIGURAÇÃO DO MODELO ───
print("🔄 Carregando Stable Diffusion 2.1...")
model_id = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
)

# Usar scheduler mais eficiente
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# GPU se disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = pipe.to(device)

print(f"✅ Modelo carregado em: {device}")

# ─── 2. GERAÇÃO BÁSICA ───
def generate_image(prompt, negative_prompt="", num_steps=25, guidance=7.5, seed=None):
    """
    Gera uma imagem a partir de um prompt textual

    Args:
        prompt: Descrição da imagem desejada
        negative_prompt: O que evitar na imagem
        num_steps: Número de steps de difusão (maior = melhor qualidade)
        guidance: Força do prompt (7-15 recomendado)
        seed: Seed para reprodutibilidade
    """
    generator = None if seed is None else torch.Generator(device).manual_seed(seed)

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance,
        generator=generator,
        height=512,
        width=512
    ).images[0]

    return image

# ─── 3. EXEMPLOS DE PROMPTS ───

# Exemplo 1: Paisagem sci-fi
prompt1 = """
futuristic cyberpunk city at sunset, neon lights, flying cars,
highly detailed, digital art, 8k, trending on artstation
"""
negative1 = "blurry, low quality, distorted"

image1 = generate_image(prompt1, negative1, num_steps=30, guidance=8.0, seed=42)
image1.save("cyberpunk_city.png")
print("✅ Imagem 1 salva: cyberpunk_city.png")

# Exemplo 2: Retrato artístico
prompt2 = """
portrait of a wise old wizard with long beard, magical glow,
fantasy art style, dramatic lighting, intricate details, oil painting
"""
negative2 = "cartoon, anime, low quality, blurry"

image2 = generate_image(prompt2, negative2, num_steps=25, guidance=7.5, seed=123)
image2.save("wizard_portrait.png")
print("✅ Imagem 2 salva: wizard_portrait.png")

# Exemplo 3: Conceito abstrato
prompt3 = """
abstract representation of artificial intelligence,
geometric shapes, blue and purple colors, minimalist design,
digital art, high resolution
"""
negative3 = "realistic, photographic, cluttered"

image3 = generate_image(prompt3, negative3, num_steps=20, guidance=6.0, seed=456)
image3.save("ai_abstract.png")
print("✅ Imagem 3 salva: ai_abstract.png")

# ─── 4. BATCH GENERATION - VARIAÇÕES ───
print("\n🎲 Gerando variações da mesma cena...")

base_prompt = "a cozy coffee shop interior, warm lighting, autumn vibes"
variations = []

for i in range(4):
    img = generate_image(base_prompt, num_steps=20, seed=1000+i)
    variations.append(img)
    print(f"  ✓ Variação {i+1}/4 gerada")

# Criar grid de variações
def create_grid(images, rows=2, cols=2):
    w, h = images[0].size
    grid = Image.new('RGB', (cols*w, rows*h))

    for idx, img in enumerate(images):
        grid.paste(img, ((idx % cols)*w, (idx // cols)*h))

    return grid

grid = create_grid(variations)
grid.save("coffee_shop_variations.png")
print("✅ Grid de variações salvo: coffee_shop_variations.png")

# ─── 5. IMG2IMG - TRANSFORMAÇÃO DE IMAGENS ───
from diffusers import StableDiffusionImg2ImgPipeline

print("\n🖼️ Preparando pipeline Img2Img...")
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None
).to(device)

# Carregar imagem de referência (use uma das geradas ou sua própria)
init_image = Image.open("cyberpunk_city.png").convert("RGB")
init_image = init_image.resize((512, 512))

# Transformar para estilo diferente
transformation_prompt = "same scene but in watercolor painting style, artistic, dreamy"
transformed = img2img_pipe(
    prompt=transformation_prompt,
    image=init_image,
    strength=0.75,  # 0 = igual original, 1 = totalmente novo
    guidance_scale=7.5,
    num_inference_steps=30
).images[0]

transformed.save("cyberpunk_watercolor.png")
print("✅ Transformação salva: cyberpunk_watercolor.png")

# ─── 6. INPAINTING - EDIÇÃO DE REGIÕES ───
from diffusers import StableDiffusionInpaintPipeline

print("\n✏️ Preparando pipeline Inpainting...")
inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16
).to(device)

# Criar máscara simples (na prática, use editor de imagem)
mask = Image.new('L', (512, 512), 0)
from PIL import ImageDraw
draw = ImageDraw.Draw(mask)
draw.rectangle([200, 200, 400, 400], fill=255)  # Área branca = será repintada

# Repintar região
inpaint_prompt = "a glowing magical portal"
inpainted = inpaint_pipe(
    prompt=inpaint_prompt,
    image=init_image,
    mask_image=mask,
    num_inference_steps=25
).images[0]

inpainted.save("inpainted_portal.png")
print("✅ Inpainting salvo: inpainted_portal.png")

print("\n" + "="*70)
print("✅ EXERCÍCIO CONCLUÍDO!")
print("="*70)
print("\n📁 Arquivos gerados:")
print("  - cyberpunk_city.png")
print("  - wizard_portrait.png")
print("  - ai_abstract.png")
print("  - coffee_shop_variations.png (grid 2x2)")
print("  - cyberpunk_watercolor.png (img2img)")
print("  - inpainted_portal.png (inpainting)")
