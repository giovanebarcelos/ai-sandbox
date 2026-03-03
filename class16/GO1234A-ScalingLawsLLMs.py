# GO1234A-34aScalingLawsMaiorÉSempre
import numpy as np

def estimate_training_compute(model_params, tokens, flops_per_token=6):
    """
    Estima FLOPs necessários para treinar LLM

    Args:
        model_params: Número de parâmetros do modelo (ex: 70e9 = 70B)
        tokens: Número de tokens de treinamento (ex: 1.4e12 = 1.4T)
        flops_per_token: ~6 FLOPs por parâmetro por token (regra geral)

    Returns:
        FLOPs totais, dias de treinamento estimados
    """
    total_flops = flops_per_token * model_params * tokens

    # Estimar tempo com hardware típico
    # 1 GPU A100: ~312 TFLOPS
    # Cluster típico: 1000 GPUs = 312 PFLOPS
    cluster_flops_per_sec = 312e15 * 1000  # 312 PFLOPS

    seconds = total_flops / cluster_flops_per_sec
    days = seconds / 86400

    # Custo estimado (GPU cloud)
    cost_per_gpu_hour = 3  # USD
    total_hours = seconds / 3600
    total_cost = total_hours * 1000 * cost_per_gpu_hour

    return {
        'flops': total_flops,
        'days': days,
        'cost_usd': total_cost
    }

# Exemplos


if __name__ == "__main__":
    print("📊 SCALING LAW CALCULATOR\n")

    # GPT-3
    gpt3 = estimate_training_compute(175e9, 300e9)
    print("GPT-3 (175B, 300B tokens):")
    print(f"  FLOPs: {gpt3['flops']:.2e}")
    print(f"  Tempo: {gpt3['days']:.1f} dias")
    print(f"  Custo: ${gpt3['cost_usd']/1e6:.1f}M\n")

    # Llama 2 70B
    llama2 = estimate_training_compute(70e9, 2e12)
    print("Llama 2 (70B, 2T tokens):")
    print(f"  FLOPs: {llama2['flops']:.2e}")
    print(f"  Tempo: {llama2['days']:.1f} dias")
    print(f"  Custo: ${llama2['cost_usd']/1e6:.1f}M\n")

    # Seu modelo customizado
    custom_params = 7e9  # 7B params
    custom_tokens = 100e9  # 100B tokens
    custom = estimate_training_compute(custom_params, custom_tokens)
    print("Seu Modelo (7B, 100B tokens):")
    print(f"  FLOPs: {custom['flops']:.2e}")
    print(f"  Tempo: {custom['days']:.1f} dias")
    print(f"  Custo: ${custom['cost_usd']/1e3:.1f}K")

    # Chinchilla optimal ratio
    def chinchilla_optimal(compute_budget_flops):
        """
        Calcula params e tokens ótimos para compute budget
        Baseado em Chinchilla Scaling Laws
        """
        # N_opt = C^0.5 / 6  (aproximação)
        # D_opt = 20 * N_opt
        params_opt = (compute_budget_flops / 6) ** 0.5
        tokens_opt = 20 * params_opt

        return params_opt, tokens_opt

    print("\n📐 CHINCHILLA OPTIMAL ALLOCATION:")
    budget = 1e24  # 1e24 FLOPs
    params, tokens = chinchilla_optimal(budget)
    print(f"Compute Budget: {budget:.2e} FLOPs")
    print(f"  → Optimal params: {params/1e9:.1f}B")
    print(f"  → Optimal tokens: {tokens/1e9:.0f}B")
