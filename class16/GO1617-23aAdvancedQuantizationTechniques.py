# GO1617-23aAdvancedQuantizationTechniques
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import matplotlib.pyplot as plt
from typing import Dict, List
import time

class QuantizationComparison:
    """
    Compara técnicas de quantização para LLMs

    Methods:
    - FP32: Full precision (baseline)
    - FP16: Half precision
    - INT8: 8-bit quantization
    - INT4: 4-bit quantization (NF4, QLoRA)
    - GPTQ: Post-training quantization

    Trade-offs: Size, Speed, Accuracy
    """

    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_model_fp32(self):
        """Load full precision model (baseline)"""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32
        )
        return model

    def load_model_fp16(self):
        """Load half precision model"""
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16
        )
        return model

    def load_model_8bit(self):
        """Load 8-bit quantized model"""
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        return model

    def load_model_4bit(self):
        """Load 4-bit quantized model (NF4)"""
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"  # NormalFloat4
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        return model

    def measure_model_size(self, model) -> float:
        """Calculate model size in MB"""
        total_size = 0
        for param in model.parameters():
            total_size += param.nelement() * param.element_size()

        size_mb = total_size / (1024 ** 2)
        return size_mb

    def measure_inference_speed(self, model, num_runs=10) -> float:
        """Measure average inference latency"""
        prompt = "The future of artificial intelligence is"
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Warmup
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=20)

        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=20)

        avg_time = (time.time() - start_time) / num_runs
        return avg_time * 1000  # Convert to ms

    def simulate_perplexity(self, quantization_type: str) -> float:
        """
        Simulate perplexity for different quantizations

        Lower is better (real values would require test set)
        """
        perplexities = {
            'FP32': 25.0,
            'FP16': 25.1,
            'INT8': 25.5,
            'INT4': 27.2,
            'GPTQ-4bit': 26.5,
        }

        return perplexities.get(quantization_type, 30.0)

    def compare_all_methods(self) -> Dict:
        """Compare all quantization methods"""

        print("🔢 Comparing Quantization Methods...\n")

        results = {}

        # FP32 (baseline)
        print("   Loading FP32...")
        model_fp32 = self.load_model_fp32()
        results['FP32'] = {
            'size_mb': self.measure_model_size(model_fp32),
            'perplexity': self.simulate_perplexity('FP32'),
            'speed_ms': 45.0,  # Simulated
            'memory_gb': 0.5
        }

        # FP16
        print("   Loading FP16...")
        model_fp16 = self.load_model_fp16()
        results['FP16'] = {
            'size_mb': self.measure_model_size(model_fp16),
            'perplexity': self.simulate_perplexity('FP16'),
            'speed_ms': 32.0,  # Simulated (faster)
            'memory_gb': 0.25
        }

        # Note: 8-bit and 4-bit require bitsandbytes library
        # Simulated results for demo
        results['INT8'] = {
            'size_mb': results['FP32']['size_mb'] / 4,
            'perplexity': self.simulate_perplexity('INT8'),
            'speed_ms': 28.0,
            'memory_gb': 0.125
        }

        results['INT4 (NF4)'] = {
            'size_mb': results['FP32']['size_mb'] / 8,
            'perplexity': self.simulate_perplexity('INT4'),
            'speed_ms': 25.0,
            'memory_gb': 0.0625
        }

        results['GPTQ-4bit'] = {
            'size_mb': results['FP32']['size_mb'] / 8,
            'perplexity': self.simulate_perplexity('GPTQ-4bit'),
            'speed_ms': 22.0,  # GPTQ optimized for speed
            'memory_gb': 0.0625
        }

        return results

# === DEMO ===

print("🔢 Advanced Quantization Techniques\n")
print("="*70)

comparator = QuantizationComparison("gpt2")

# Run comparison
results = comparator.compare_all_methods()

print("\n📊 RESULTS:\n")

print(f"{'Method':<15} {'Size (MB)':<12} {'Perplexity':<12} {'Speed (ms)':<12} {'Memory (GB)'}")
print("-" * 70)

for method, metrics in results.items():
    print(f"{method:<15} {metrics['size_mb']:<12.1f} {metrics['perplexity']:<12.1f} "
          f"{metrics['speed_ms']:<12.1f} {metrics['memory_gb']:<12.3f}")

# Calculate compression ratios
fp32_size = results['FP32']['size_mb']
print("\n📌 Compression Ratios (vs FP32):\n")
for method, metrics in results.items():
    ratio = fp32_size / metrics['size_mb']
    quality_loss = ((metrics['perplexity'] - results['FP32']['perplexity']) / 
                   results['FP32']['perplexity'] * 100)

    print(f"{method:<15} {ratio:.1f}x smaller, {quality_loss:+.1f}% perplexity change")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

methods = list(results.keys())

# 1. Model size comparison
ax = axes[0, 0]
sizes = [results[m]['size_mb'] for m in methods]
colors_size = ['red', 'orange', 'yellow', 'lightgreen', 'green']

bars = ax.bar(methods, sizes, color=colors_size, alpha=0.7)
ax.set_ylabel('Model Size (MB)')
ax.set_title('Model Size by Quantization Method')
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

for bar, size in zip(bars, sizes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{size:.0f}', ha='center', va='bottom', fontweight='bold')

# 2. Quality vs Size tradeoff
ax = axes[0, 1]
sizes_plot = [results[m]['size_mb'] for m in methods]
perplexities = [results[m]['perplexity'] for m in methods]

scatter = ax.scatter(sizes_plot, perplexities, s=200, alpha=0.6, c=colors_size)

for i, method in enumerate(methods):
    ax.annotate(method, (sizes_plot[i], perplexities[i]), 
               xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('Model Size (MB)')
ax.set_ylabel('Perplexity (lower = better)')
ax.set_title('Quality vs Size Trade-off')
ax.grid(alpha=0.3)
ax.invert_xaxis()

# Add Pareto frontier
ax.plot([max(sizes_plot), min(sizes_plot)], 
       [min(perplexities), max(perplexities)], 
       'r--', alpha=0.3, linewidth=2, label='Pareto Frontier')
ax.legend()

# 3. Inference speed comparison
ax = axes[1, 0]
speeds = [results[m]['speed_ms'] for m in methods]

bars = ax.barh(methods, speeds, color=colors_size[::-1], alpha=0.7)
ax.set_xlabel('Latency (ms)')
ax.set_title('Inference Speed (lower = faster)')
ax.grid(axis='x', alpha=0.3)

for bar, speed in zip(bars, speeds):
    width = bar.get_width()
    ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
            f'{speed:.0f}ms', ha='left', va='center', fontweight='bold')

# 4. Memory usage
ax = axes[1, 1]
memory = [results[m]['memory_gb'] for m in methods]

bars = ax.bar(methods, memory, color=colors_size, alpha=0.7)
ax.set_ylabel('Memory Usage (GB)')
ax.set_title('Runtime Memory Consumption')
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)

for bar, mem in zip(bars, memory):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{mem:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig('quantization_comparison.png', dpi=150, bbox_inches='tight')
print("\n📊 Gráfico salvo: quantization_comparison.png")

print("\n✅ Quantization comparison implementado!")
print("\n💡 RECOMMENDATIONS:")
print("   - FP16: Default for modern GPUs (free speedup)")
print("   - INT8: 4x compression, <1% quality loss")
print("   - INT4 (NF4): 8x compression, ~2-3% quality loss")
print("   - GPTQ: Best for inference speed")
print("   - Use 4-bit for large models on consumer hardware")
print("\n💡 WHEN TO USE EACH:")
print("   FP32: Baseline, research")
print("   FP16: Production (GPU), minimal trade-offs")
print("   INT8: CPU inference, mobile")
print("   INT4: Consumer GPU (24GB → fit 70B model)")
print("   GPTQ/GGUF: Optimized for specific hardware")
