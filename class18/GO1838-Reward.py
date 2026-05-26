"""
GO1838 - Recompensa para Sistema de Recomendação de E-Commerce
==============================================================
Demonstra a função de recompensa para RL em recomendação de produtos.
Requer apenas numpy.

Contexto:
  Sites de e-commerce (Amazon, Shopee, Mercado Livre) usam RL para aprender
  qual produto recomendar a cada usuário, em cada momento, para maximizar
  receita enquanto mantém uma boa experiência de compra.

  O agente precisa equilibrar:
  - Engajamento imediato (cliques) — sinal fraco mas frequente
  - Intenção de compra (adicionar ao carrinho) — sinal médio
  - Conversão (compra efetiva) — sinal forte mas raro
  - Valor do pedido — nem sempre maximizar é o melhor para o cliente
  - Evitar bounce (usuário sai sem interagir) — sinal negativo
"""

import numpy as np
from typing import Optional


def calcular_reward_ecommerce(
    comprou: bool,
    adicionou_carrinho: bool,
    clicou: bool,
    bounce_rate_alto: bool,
    valor_pedido: float = 0.0,
    # Pesos — podem ser ajustados por A/B testing
    peso_compra: float = 10.0,
    peso_carrinho: float = 1.0,
    peso_clique: float = 0.1,
    penalidade_bounce: float = 0.5,
    peso_valor: float = 5.0,
) -> dict:
    """
    Calcula recompensa multi-objetivo para recomendação e-commerce.

    Hierarquia de sinais:
      compra > carrinho > clique > bounce (negativo)

    valor_pedido: normalizado (ex: valor / 100.0) para não dominar a reward
    """
    r_compra = peso_compra if comprou else 0.0
    r_carrinho = peso_carrinho if adicionou_carrinho else 0.0
    r_clique = peso_clique if clicou else 0.0
    r_bounce = -penalidade_bounce if bounce_rate_alto else 0.0
    r_valor = peso_valor * (valor_pedido / 100.0) if comprou else 0.0

    total = r_compra + r_carrinho + r_clique + r_bounce + r_valor

    return {
        "compra": r_compra,
        "carrinho": r_carrinho,
        "clique": r_clique,
        "bounce": r_bounce,
        "valor_pedido": r_valor,
        "total": total,
    }


def simular_funil_recomendacao(
    n_usuarios: int = 20,
    seed: int = 42,
) -> None:
    """
    Simula um funil de recomendação com taxas reais de e-commerce.

    Taxas típicas de conversão:
      - CTR (click-through rate): 2-5%
      - Add-to-cart rate: 8-12% dos que clicam
      - Purchase rate: 3-5% dos que adicionam ao carrinho
      - Bounce rate: 40-60% das sessões
    """
    np.random.seed(seed)

    total_reward = 0.0
    compras = 0
    carrinhos = 0
    cliques = 0
    bounces = 0

    print(f"  Simulando {n_usuarios} recomendações...\n")
    print(f"  {'Usuário':>8} | {'Clicou':>7} | {'Carrinho':>9} | "
          f"{'Comprou':>8} | {'Valor':>8} | {'Reward':>8}")
    print("  " + "-" * 58)

    for i in range(1, n_usuarios + 1):
        # Funil de conversão com probabilidades realistas
        clicou = np.random.random() < 0.35         # CTR 35%
        adicionou = clicou and (np.random.random() < 0.20)   # 20% dos cliques
        comprou = adicionou and (np.random.random() < 0.40)  # 40% dos carrinhos
        bounce = not clicou and (np.random.random() < 0.60)  # 60% de bounce se não clicou
        valor = np.random.uniform(50, 500) if comprou else 0.0

        comp = calcular_reward_ecommerce(
            comprou=comprou,
            adicionou_carrinho=adicionou,
            clicou=clicou,
            bounce_rate_alto=bounce,
            valor_pedido=valor,
        )
        total_reward += comp["total"]
        if comprou:
            compras += 1
        if adicionou:
            carrinhos += 1
        if clicou:
            cliques += 1
        if bounce:
            bounces += 1

        # Exibir apenas os 8 primeiros para não poluir a saída
        if i <= 8:
            c_str = "sim" if clicou else "nao"
            a_str = "sim" if adicionou else "nao"
            p_str = "SIM" if comprou else "nao"
            v_str = f"R${valor:.0f}" if comprou else "—"
            print(f"  {i:>8} | {c_str:>7} | {a_str:>9} | "
                  f"{p_str:>8} | {v_str:>8} | {comp['total']:>+8.3f}")

    if n_usuarios > 8:
        print(f"  ... ({n_usuarios - 8} usuários adicionais omitidos)")

    print(f"\n  RESUMO ({n_usuarios} usuários):")
    print(f"    Cliques:     {cliques:>3}  ({cliques/n_usuarios*100:.0f}%)")
    print(f"    Carrinhos:   {carrinhos:>3}  ({carrinhos/n_usuarios*100:.0f}%)")
    print(f"    Compras:     {compras:>3}  ({compras/n_usuarios*100:.0f}%)")
    print(f"    Bounces:     {bounces:>3}  ({bounces/n_usuarios*100:.0f}%)")
    print(f"    Reward total: {total_reward:>+8.2f}")
    print(f"    Reward médio: {total_reward/n_usuarios:>+8.3f} por usuário")


if __name__ == "__main__":
    print("=" * 60)
    print("GO1838 - RECOMPENSA PARA RECOMENDACAO E-COMMERCE")
    print("=" * 60)

    print("\nFORMULA:")
    print()
    print("  reward = + 10  if comprou            # Conversao!")
    print("           + 1   if adicionou_carrinho  # Intencao de compra")
    print("           + 0.1 if clicou              # Curiosidade")
    print("           - 0.5 if bounce_rate_alto    # Saiu sem interagir")
    print("           + 5 * (valor_pedido / 100)   # Valor monetario")

    print()
    print("─" * 60)
    print("CENARIOS:")
    print("─" * 60)

    cenarios = [
        # (descricao, comprou, carrinho, clicou, bounce, valor)
        ("Compra alta valor (R$500)",      True,  True,  True,  False, 500.0),
        ("Compra baixo valor (R$30)",      True,  True,  True,  False, 30.0),
        ("Adicionou ao carrinho, nao comprou", False, True,  True,  False, 0.0),
        ("Apenas clicou",                  False, False, True,  False, 0.0),
        ("Bounce (saiu sem interagir)",    False, False, False, True,  0.0),
        ("Sem interacao alguma",           False, False, False, False, 0.0),
    ]

    for desc, comp, cart, cli, bou, val in cenarios:
        r = calcular_reward_ecommerce(comp, cart, cli, bou, val)
        print(f"\n  [{desc}]")
        for k, v in r.items():
            if k != "total":
                print(f"    {k:<20}: {v:>+7.2f}")
        print(f"    {'TOTAL':<20}: {r['total']:>+7.2f}")

    print()
    print("─" * 60)
    print("SIMULACAO FUNIL (20 USUARIOS):")
    print("─" * 60)
    print()
    simular_funil_recomendacao(n_usuarios=20)

    print()
    print("─" * 60)
    print("DESAFIOS DO RL EM E-COMMERCE:")
    print("─" * 60)
    print("  + Feedback imediato (clique em segundos)")
    print("  + Bilhões de interações por dia para treinar")
    print("  - Sparse rewards: compras são raras vs visualizações")
    print("  - Filter bubble: agente pode recomendar sempre o mesmo")
    print("  - Exploration vs exploitation: recomendar novos produtos?")
    print("  - Cold start: usuários novos sem histórico")
    print("  - Ética: não maximizar tempo em tela a qualquer custo")
