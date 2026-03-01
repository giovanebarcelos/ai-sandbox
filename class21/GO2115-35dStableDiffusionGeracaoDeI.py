# GO2115-35dStableDiffusionGeraçãoDeI
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

print("="*70)
print("STABLE DIFFUSION - Geração de Imagens por Texto")
print("="*70)

# Verificar GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nDispositivo: {device}")

if device == "cpu":
    print("⚠️  AVISO: Stable Diffusion é MUITO lenta em CPU!")
    print("   Recomendado: GPU com pelo menos 8GB VRAM")

# 1. CARREGAR MODELO
print("\n📥 Carregando Stable Diffusion v1.5...")
print("   (Primeira vez: ~4GB de download)\n")

model_id = "runwayml/stable-diffusion-v1-5"

# Para CPU (mais lento)
if device == "cpu":
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float32
    )
else:
    # Para GPU (rápido)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )

pipe = pipe.to(device)

# Otimizações
if device == "cuda":
    pipe.enable_attention_slicing()  # Reduz uso de memória
    # pipe.enable_xformers_memory_efficient_attention()  # Mais rápido (requer xformers)

print("✅ Modelo carregado!")

# 2. GERAR IMAGENS COM DIFERENTES PROMPTS
prompts = [
    "a beautiful sunset over mountains, oil painting style, highly detailed",
    "a futuristic city with flying cars, cyberpunk, neon lights, 4k",
    "a cute robot playing with a cat, cartoon style, colorful",
    "an astronaut riding a horse on mars, photorealistic"
]

print("\n🎨 Gerando imagens...\n")

generated_images = []

for i, prompt in enumerate(prompts, 1):
    print(f"[{i}/{len(prompts)}] Prompt: {prompt}")

    # Gerar
    with torch.autocast(device):
        image = pipe(
            prompt,
            num_inference_steps=50,  # Mais steps = melhor qualidade (mais lento)
            guidance_scale=7.5,      # Quanto seguir o prompt (7-9 é bom)
            height=512,
            width=512
        ).images[0]

    generated_images.append((prompt, image))

    # Salvar individual
    image.save(f'sd_image_{i}.png')
    print(f"  ✅ Salva: sd_image_{i}.png\n")

# 3. VISUALIZAR TODAS
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.ravel()

for idx, (prompt, img) in enumerate(generated_images):
    axes[idx].imshow(img)
    axes[idx].set_title(prompt, fontsize=10, wrap=True)
    axes[idx].axis('off')

plt.suptitle('Stable Diffusion - Imagens Geradas', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('stable_diffusion_grid.png', dpi=150, bbox_inches='tight')
print("✅ Grid salvo: stable_diffusion_grid.png")

# 4. VARIAÇÕES COM SEEDS
print("\n🌱 Gerando variações com diferentes seeds...\n")

prompt_var = "a magical forest with glowing mushrooms, fantasy art"
seeds = [42, 123, 456, 789]

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

for idx, seed in enumerate(seeds):
    generator = torch.Generator(device=device).manual_seed(seed)

    with torch.autocast(device):
        image = pipe(
            prompt_var,
            num_inference_steps=50,
            guidance_scale=7.5,
            generator=generator
        ).images[0]

    axes[idx].imshow(image)
    axes[idx].set_title(f'Seed: {seed}', fontsize=12)
    axes[idx].axis('off')

    print(f"  Seed {seed} ✅")

plt.suptitle(f'Variações: "{prompt_var}"', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sd_variations.png', dpi=150, bbox_inches='tight')
print("\n✅ Variações salvas: sd_variations.png")

# 5. NEGATIVE PROMPTS
print("\n🚫 Testando Negative Prompts...\n")

prompt_pos = "a beautiful portrait of a woman, professional photography"
negative_prompt = "ugly, blurry, low quality, distorted, disfigured"

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Sem negative prompt
with torch.autocast(device):
    img_without = pipe(
        prompt_pos,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]

axes[0].imshow(img_without)
axes[0].set_title('Sem Negative Prompt', fontsize=12, fontweight='bold')
axes[0].axis('off')

# Com negative prompt
with torch.autocast(device):
    img_with = pipe(
        prompt_pos,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        guidance_scale=7.5
    ).images[0]

axes[1].imshow(img_with)
axes[1].set_title('Com Negative Prompt', fontsize=12, fontweight='bold')
axes[1].axis('off')

plt.suptitle('Impacto do Negative Prompt', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sd_negative_prompt.png', dpi=150, bbox_inches='tight')
print("✅ Comparação salva: sd_negative_prompt.png")

print("\n" + "="*70)
print("📊 STABLE DIFFUSION - RESUMO")
print("="*70)
print("\n🎯 Hiperparâmetros Importantes:")
print("  - num_inference_steps: 20-50 (qualidade vs velocidade)")
print("  - guidance_scale: 7-9 (quanto seguir o prompt)")
print("  - seed: Reprodutibilidade")
print("  - negative_prompt: O que NÃO gerar")

print("\n💡 Dicas para Bons Prompts:")
print("  ✓ Seja específico e descritivo")
print("  ✓ Mencione estilo artístico (oil painting, photorealistic, etc.)")
print("  ✓ Adicione qualificadores (highly detailed, 4k, professional)")
print("  ✓ Use negative prompts para evitar problemas comuns")
print("  ✓ Experimente com diferentes seeds")

print("\n🚀 Modelos Alternativos:")
print("  - SDXL: Stable Diffusion XL (melhor qualidade)")
print("  - DALL-E 3: OpenAI (via API)")
print("  - Midjourney: Via Discord (melhor qualidade artística)")
print("  - Imagen: Google (research only)")

print("\n✅ Geração de imagens completa!")
