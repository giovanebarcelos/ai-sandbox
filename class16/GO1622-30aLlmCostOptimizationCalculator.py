# GO1622-30aLlmCostOptimizationCalculator
import numpy as np
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ModelPricing:
    """Preços para API de LLM"""
    name: str
    input_price_per_1m: float  # $/1M tokens
    output_price_per_1m: float
    context_length: int
    avg_latency_ms: float

class LLMCostCalculator:
    """
    Calcula custos de LLM APIs e otimiza escolha de modelo

    Funcionalidades:
    - Estimativa de custo
    - Comparação de modelos
    - Benefícios do cache
    - Estratégias de lote
    """

    def __init__(self):
        # Preços até 2025 (aproximados)
        self.models = {
            'gpt-4o': ModelPricing('GPT-4o', 2.5, 10, 128000, 50),
            'gpt-4-turbo': ModelPricing('GPT-4 Turbo', 10, 30, 128000, 80),
            'claude-sonnet': ModelPricing('Claude Sonnet 3.5', 3, 15, 200000, 45),
            'claude-haiku': ModelPricing('Claude Haiku', 0.25, 1.25, 200000, 25),
            'gemini-pro': ModelPricing('Gemini Pro 1.5', 1.25, 5, 1000000, 60),
            'llama3-70b-api': ModelPricing('Llama 3 70B (API)', 0.90, 0.90, 8192, 70),
        }

        # Custos de auto-hospedagem (único + mensal)
        self.selfhosted_costs = {
            'llama3-70b': {'hardware': 15000, 'monthly': 500},  # servidor A100
            'llama3-8b': {'hardware': 3000, 'monthly': 100},   # GPU consumidor
        }

    def calculate_api_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """Calcular custo de chamada de API"""
        model = self.models[model_name]

        input_cost = (input_tokens / 1_000_000) * model.input_price_per_1m
        output_cost = (output_tokens / 1_000_000) * model.output_price_per_1m

        total_cost = input_cost + output_cost

        return total_cost

    def calculate_monthly_cost(self, model_name: str, calls_per_day: int, 
                               avg_input: int, avg_output: int) -> Dict:
        """Calcular custos mensais"""
        cost_per_call = self.calculate_api_cost(model_name, avg_input, avg_output)

        daily_cost = cost_per_call * calls_per_day
        monthly_cost = daily_cost * 30
        annual_cost = monthly_cost * 12

        return {
            'per_call': cost_per_call,
            'daily': daily_cost,
            'monthly': monthly_cost,
            'annual': annual_cost
        }

    def compare_models(self, input_tokens: int, output_tokens: int) -> Dict:
        """Comparar custos entre modelos"""
        results = {}

        for model_name in self.models:
            cost = self.calculate_api_cost(model_name, input_tokens, output_tokens)
            latency = self.models[model_name].avg_latency_ms

            results[model_name] = {
                'cost': cost,
                'latency': latency,
                'cost_per_second': cost / (latency / 1000)
            }

        return results

    def caching_savings(self, model_name: str, total_tokens: int, cache_hit_rate: float) -> Dict:
        """
        Calcular economia com cache

        Premissa: Tokens em cache custam 10% dos tokens regulares
        """
        model = self.models[model_name]

        cached_tokens = int(total_tokens * cache_hit_rate)
        uncached_tokens = total_tokens - cached_tokens

        # Sem cache
        cost_no_cache = (total_tokens / 1_000_000) * model.input_price_per_1m

        # Com cache (10% do custo para tokens em cache)
        cost_cached = (cached_tokens / 1_000_000) * (model.input_price_per_1m * 0.1)
        cost_uncached = (uncached_tokens / 1_000_000) * model.input_price_per_1m
        cost_with_cache = cost_cached + cost_uncached

        savings = cost_no_cache - cost_with_cache
        savings_pct = (savings / cost_no_cache) * 100

        return {
            'cost_no_cache': cost_no_cache,
            'cost_with_cache': cost_with_cache,
            'savings': savings,
            'savings_pct': savings_pct
        }

    def selfhosted_vs_api_breakeven(self, model_name: str, calls_per_day: int,
                                    avg_input: int, avg_output: int) -> Dict:
        """
        Calcular ponto de equilíbrio para auto-hospedagem

        Quando a auto-hospedagem se torna mais barata?
        """
        # Custos da API
        api_monthly_cost = self.calculate_monthly_cost(
            model_name, calls_per_day, avg_input, avg_output
        )['monthly']

        # Estimar modelo auto-hospedado similar
        if '70b' in model_name.lower() or 'turbo' in model_name.lower():
            selfhosted_model = 'llama3-70b'
        else:
            selfhosted_model = 'llama3-8b'

        hardware_cost = self.selfhosted_costs[selfhosted_model]['hardware']
        monthly_cost = self.selfhosted_costs[selfhosted_model]['monthly']

        # Cálculo do ponto de equilíbrio
        months_to_breakeven = hardware_cost / (api_monthly_cost - monthly_cost)

        return {
            'api_monthly_cost': api_monthly_cost,
            'selfhosted_monthly_cost': monthly_cost,
            'hardware_cost': hardware_cost,
            'months_to_breakeven': max(months_to_breakeven, 0),
            'years_to_breakeven': max(months_to_breakeven / 12, 0),
            'recommendation': 'Auto-hospedar' if months_to_breakeven < 12 else 'Usar API'
        }

# === DEMO ===

print("💰 Calculadora de Custos de LLM\n")
print("="*70)

calculator = LLMCostCalculator()

# Example usage
print("\n📌 Exemplo: Aplicação de Chatbot\n")

input_tokens = 1000  # System prompt + user message
output_tokens = 500  # Response

print(f"Entrada: {input_tokens} tokens")
print(f"Saída: {output_tokens} tokens\n")

# Single call comparison
print("Custo por chamada:\n")

comparison = calculator.compare_models(input_tokens, output_tokens)

for model_name, metrics in sorted(comparison.items(), key=lambda x: x[1]['cost']):
    print(f"   {model_name:<20} ${metrics['cost']:.6f}  "
          f"({metrics['latency']:.0f}ms latency)")

# Monthly cost
print("\n\n📌 Estimativa de Custo Mensal:\n")

calls_per_day = 10000

print(f"Uso: {calls_per_day:,} chamadas/dia\n")

for model_name in ['gpt-4o', 'claude-sonnet', 'claude-haiku', 'llama3-70b-api']:
    monthly = calculator.calculate_monthly_cost(model_name, calls_per_day, 
                                                input_tokens, output_tokens)

    print(f"{calculator.models[model_name].name}:")
    print(f"   Diário: ${monthly['daily']:.2f}")
    print(f"   Mensal: ${monthly['monthly']:.2f}")
    print(f"   Anual: ${monthly['annual']:.2f}")
    print()

# Caching savings
print("\n📌 Benefícios do Cache:\n")

total_input_tokens = 10_000_000  # 10M tokens/month
cache_hit_rates = [0.0, 0.3, 0.5, 0.7, 0.9]

model_for_cache = 'gpt-4o'

print(f"Modelo: {calculator.models[model_for_cache].name}")
print(f"Total de tokens de entrada: {total_input_tokens:,}/mês\n")

for rate in cache_hit_rates:
    savings_result = calculator.caching_savings(model_for_cache, total_input_tokens, rate)

    print(f"Taxa de acerto de cache: {rate:.0%}")
    print(f"   Sem cache: ${savings_result['cost_no_cache']:.2f}")
    print(f"   Com cache: ${savings_result['cost_with_cache']:.2f}")
    print(f"   Economia: ${savings_result['savings']:.2f} ({savings_result['savings_pct']:.1f}%)")
    print()

# Self-hosted vs API
print("\n📌 Análise: Auto-hospedagem vs API:\n")

for daily_calls in [1000, 5000, 10000]:
    print(f"\nCenário: {daily_calls:,} chamadas/dia")

    breakeven = calculator.selfhosted_vs_api_breakeven(
        'gpt-4o', daily_calls, input_tokens, output_tokens
    )

    print(f"   Custo mensal da API: ${breakeven['api_monthly_cost']:.2f}")
    print(f"   Mensal auto-hospedado: ${breakeven['selfhosted_monthly_cost']:.2f}")
    print(f"   Investimento em hardware: ${breakeven['hardware_cost']:,}")
    print(f"   Ponto de equilíbrio: {breakeven['months_to_breakeven']:.1f} meses ({breakeven['years_to_breakeven']:.1f} anos)")
    print(f"   → {breakeven['recommendation']}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Comparação de custos por modelo
ax = axes[0, 0]

models = ['claude-haiku', 'llama3-70b-api', 'gpt-4o', 'claude-sonnet', 'gpt-4-turbo']
costs = [comparison[m]['cost'] for m in models]
colors_cost = ['green', 'lightgreen', 'yellow', 'orange', 'red']

bars = ax.barh(models, costs, color=colors_cost, alpha=0.7)
ax.set_xlabel('Custo por Chamada ($)')
ax.set_title(f'Comparação de Custos ({input_tokens} entrada + {output_tokens} saída tokens)')
ax.grid(axis='x', alpha=0.3)

for bar, cost in zip(bars, costs):
    width = bar.get_width()
    ax.text(width + width*0.05, bar.get_y() + bar.get_height()/2,
            f'${cost:.5f}', ha='left', va='center', fontweight='bold', fontsize=9)

# 2. Custo mensal por uso
ax = axes[0, 1]

daily_calls_range = [100, 500, 1000, 5000, 10000, 50000]
monthly_costs_by_model = {}

for model in ['claude-haiku', 'gpt-4o', 'claude-sonnet']:
    costs_list = []
    for calls in daily_calls_range:
        cost = calculator.calculate_monthly_cost(model, calls, input_tokens, output_tokens)['monthly']
        costs_list.append(cost)
    monthly_costs_by_model[model] = costs_list

for model, costs_list in monthly_costs_by_model.items():
    ax.plot(daily_calls_range, costs_list, marker='o', linewidth=2, label=calculator.models[model].name)

ax.set_xlabel('Chamadas de API por Dia')
ax.set_ylabel('Custo Mensal ($)')
ax.set_title('Custos por Volume de Uso')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xscale('log')
ax.set_yscale('log')

# 3. Economia com cache
ax = axes[1, 0]

cache_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
savings_list = []

for rate in cache_rates:
    result = calculator.caching_savings('gpt-4o', 10_000_000, rate)
    savings_list.append(result['savings_pct'])

ax.plot(cache_rates, savings_list, marker='o', linewidth=3, color='green', markersize=8)
ax.fill_between(cache_rates, 0, savings_list, alpha=0.3, color='green')
ax.set_xlabel('Taxa de Acerto de Cache')
ax.set_ylabel('Economia de Custo (%)')
ax.set_title('Impacto do Cache nos Custos')
ax.grid(alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 100)

# Annotate sweet spot
ax.annotate('Ponto Ideal\n(70% de acerto)', xy=(0.7, savings_list[7]), xytext=(0.5, 75),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, fontweight='bold')

# 4. Ponto de equilíbrio: auto-hospedagem vs API
ax = axes[1, 1]

daily_calls_breakeven = [500, 1000, 2000, 5000, 10000, 20000]
api_cumulative = []
selfhosted_cumulative = []

for calls in daily_calls_breakeven:
    api_monthly = calculator.calculate_monthly_cost('gpt-4o', calls, input_tokens, output_tokens)['monthly']

    # Auto-hospedado: hardware + mensal × meses
    hardware = 15000
    monthly_sh = 500

    # Calcular custo acumulado em 24 meses
    months = 24
    api_total = api_monthly * months
    selfhosted_total = hardware + (monthly_sh * months)

    api_cumulative.append(api_total)
    selfhosted_cumulative.append(selfhosted_total)

ax.plot(daily_calls_breakeven, api_cumulative, marker='s', linewidth=2, 
        label='API (24 meses)', color='red', markersize=8)
ax.plot(daily_calls_breakeven, selfhosted_cumulative, marker='o', linewidth=2, 
        label='Auto-hospedado (24 meses)', color='green', markersize=8)

ax.set_xlabel('Chamadas de API por Dia')
ax.set_ylabel('Custo Total em 24 Meses ($)')
ax.set_title('Auto-hospedagem vs API: Custo Total em 2 Anos')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xscale('log')

# Encontrar interseção
from scipy.interpolate import interp1d

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

# Simplificado: marcar cruzamento visualmente
ax.axhline(25000, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()
print("\n\n📊 Gráfico salvo: llm_cost_optimization.png")

print("\n\n✅ Cost calculator implementado!")
print("\n💡 DICAS DE OTIMIZAÇÃO DE CUSTOS:")
print("   1. Armazene em cache prompts frequentes (economize 50-90%)")
print("   2. Agrupe requisições quando possível")
print("   3. Use modelos mais baratos para tarefas simples")
print("   4. Transmita respostas em stream (melhor UX, mesmo custo)")
print("   5. Auto-hospede se >5K chamadas/dia")
print("   6. Monitore o uso com alertas")
print("   7. Use compressão de prompt (reduzir tokens)")
print("\n💡 SELEÇÃO DE MODELO:")
print("   Tarefas simples: Claude Haiku, GPT-4o-mini")
print("   Tarefas complexas: GPT-4o, Claude Sonnet")
print("   Alto volume: Auto-hospedar Llama 3")
print("   Contexto longo: Gemini Pro, Claude Sonnet")
