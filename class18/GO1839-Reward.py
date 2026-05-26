"""
GO1839 - Recompensa para Sistema de Recomendação de Conteúdo
============================================================
Demonstra a função de recompensa para RL em recomendação de conteúdo.
Requer apenas numpy.

Contexto:
  Plataformas como YouTube, TikTok, Instagram e Spotify usam RL para decidir
  qual conteúdo mostrar a cada usuário. A função de recompensa define o que o
  agente otimiza — e isso tem profundas consequências sociais.

  YouTube (2016): ao otimizar apenas "tempo assistido", o algoritmo passou a
  recomendar conteúdo cada vez mais extremo (filter bubble, radicalização).
  A solução foi adicionar penalidades por toxicidade e desinformação.

Componentes da recompensa:
  + Engajamento: likes, comentários, shares, tempo assistido
  + Diversidade: mostrar fontes e perspectivas variadas
  - Toxicidade: evitar ódio, violência, assédio
  - Desinformação: penalizar fake news e conteúdo enganoso
  + Time well spent: usuário se sente bem após usar a plataforma
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class InteracaoUsuario:
    """Registro de uma interação do usuário com um conteúdo recomendado."""
    tempo_assistido_s: float      # Segundos assistidos
    duracao_total_s: float        # Duração total do conteúdo
    deu_like: bool
    compartilhou: bool
    comentou: bool
    reportou_conteudo: bool       # Sinalizou como inapropriado
    avaliacao_qualidade: float    # Survey pós-sessão (0 a 5)
    # Metadados do conteúdo (moderação)
    score_toxicidade: float       # [0, 1] — resultado de classificador
    score_desinformacao: float    # [0, 1] — resultado de fact-checker


def calcular_reward_recomendacao(
    interacao: InteracaoUsuario,
    # Pesos configuráveis
    peso_engajamento: float = 1.0,
    peso_diversidade_bonus: float = 0.5,
    peso_toxicidade: float = 10.0,
    peso_desinformacao: float = 8.0,
    peso_satisfacao: float = 2.0,
) -> dict:
    """
    Calcula recompensa multi-objetivo para recomendação de conteúdo.

    Balanceia engajamento (curto prazo) com bem-estar (longo prazo).
    Penalidades fortes para toxicidade/desinformação impedem o agente
    de recomendar conteúdo nocivo mesmo que "engaje" muito.
    """
    # Taxa de completion: quanto do conteúdo foi assistido
    completion_rate = min(1.0, interacao.tempo_assistido_s / max(1, interacao.duracao_total_s))

    # Engajamento: combinação de sinais de interação
    engajamento = (
        completion_rate * 0.5       # Completar > assistir brevemente
        + (0.3 if interacao.deu_like else 0.0)
        + (0.3 if interacao.compartilhou else 0.0)
        + (0.2 if interacao.comentou else 0.0)
        - (1.0 if interacao.reportou_conteudo else 0.0)  # Reporte = muito ruim
    )
    r_engajamento = peso_engajamento * max(-1.0, engajamento)

    # Bônus de diversidade (simulado — em produção vem do perfil do usuário)
    # Conteúdo de qualidade moderada mas de nicho diferente = bônus
    r_diversidade = peso_diversidade_bonus * (1 - interacao.score_toxicidade)

    # Penalidade de toxicidade: exponencial para casos extremos
    # score 0.5 → penalidade 5; score 0.9 → penalidade 9
    r_toxicidade = -peso_toxicidade * interacao.score_toxicidade

    # Penalidade de desinformação
    r_desinformacao = -peso_desinformacao * interacao.score_desinformacao

    # Satisfação do usuário (proxy de "time well spent")
    satisfacao_normalizada = (interacao.avaliacao_qualidade - 2.5) / 2.5
    r_satisfacao = peso_satisfacao * satisfacao_normalizada

    total = r_engajamento + r_diversidade + r_toxicidade + r_desinformacao + r_satisfacao

    return {
        "engajamento": r_engajamento,
        "diversidade_bonus": r_diversidade,
        "toxicidade_penalty": r_toxicidade,
        "desinformacao_penalty": r_desinformacao,
        "satisfacao_usuario": r_satisfacao,
        "total": total,
    }


def simular_sessao_recomendacao(n_conteudos: int = 10, seed: int = 42) -> None:
    """
    Simula uma sessão com múltiplos conteúdos recomendados.
    Inclui mix de conteúdo de qualidade, tóxico e desinformação.
    """
    np.random.seed(seed)

    total_reward = 0.0
    print(f"  {'#':>3} | {'Completion':>10} | {'Tóxico':>7} | "
          f"{'DesiInfo':>8} | {'Like':>5} | {'Reward':>8}")
    print("  " + "-" * 53)

    for i in range(1, n_conteudos + 1):
        # Simular diferentes tipos de conteúdo
        tipo = np.random.choice(["normal", "viral", "toxico", "desinformacao"],
                                p=[0.6, 0.2, 0.1, 0.1])

        if tipo == "normal":
            interacao = InteracaoUsuario(
                tempo_assistido_s=np.random.uniform(60, 300),
                duracao_total_s=300,
                deu_like=np.random.random() < 0.3,
                compartilhou=np.random.random() < 0.1,
                comentou=np.random.random() < 0.05,
                reportou_conteudo=False,
                avaliacao_qualidade=np.random.uniform(3.0, 4.5),
                score_toxicidade=np.random.uniform(0, 0.1),
                score_desinformacao=np.random.uniform(0, 0.05),
            )
        elif tipo == "viral":
            interacao = InteracaoUsuario(
                tempo_assistido_s=np.random.uniform(180, 300),
                duracao_total_s=300,
                deu_like=True,
                compartilhou=np.random.random() < 0.5,
                comentou=np.random.random() < 0.3,
                reportou_conteudo=False,
                avaliacao_qualidade=np.random.uniform(4.0, 5.0),
                score_toxicidade=np.random.uniform(0, 0.15),
                score_desinformacao=np.random.uniform(0, 0.1),
            )
        elif tipo == "toxico":
            interacao = InteracaoUsuario(
                tempo_assistido_s=np.random.uniform(100, 250),  # Ainda engaja...
                duracao_total_s=250,
                deu_like=np.random.random() < 0.4,
                compartilhou=np.random.random() < 0.2,
                comentou=np.random.random() < 0.3,
                reportou_conteudo=np.random.random() < 0.3,
                avaliacao_qualidade=np.random.uniform(1.5, 3.0),
                score_toxicidade=np.random.uniform(0.6, 0.95),
                score_desinformacao=np.random.uniform(0, 0.3),
            )
        else:  # desinformacao
            interacao = InteracaoUsuario(
                tempo_assistido_s=np.random.uniform(80, 200),
                duracao_total_s=200,
                deu_like=np.random.random() < 0.35,
                compartilhou=np.random.random() < 0.25,  # Fake news se espalham!
                comentou=np.random.random() < 0.2,
                reportou_conteudo=np.random.random() < 0.2,
                avaliacao_qualidade=np.random.uniform(2.0, 3.5),
                score_toxicidade=np.random.uniform(0.1, 0.5),
                score_desinformacao=np.random.uniform(0.6, 0.95),
            )

        comp = calcular_reward_recomendacao(interacao)
        total_reward += comp["total"]

        completion = interacao.tempo_assistido_s / interacao.duracao_total_s
        like_str = "sim" if interacao.deu_like else "nao"
        print(f"  {i:>3} | {completion:>9.0%}  | {interacao.score_toxicidade:>6.2f} | "
              f"{interacao.score_desinformacao:>7.2f}  | {like_str:>5} | "
              f"{comp['total']:>+8.3f}")

    print(f"\n  Reward total da sessão: {total_reward:>+8.2f}")
    print(f"  Reward médio por conteúdo: {total_reward/n_conteudos:>+8.3f}")


if __name__ == "__main__":
    print("=" * 60)
    print("GO1839 - RECOMPENSA PARA RECOMENDACAO DE CONTEUDO")
    print("=" * 60)

    print("\nFORMULA:")
    print()
    print("  reward = + engagement_score      # Likes, shares, completion")
    print("           + diversity_bonus       # Mostrar fontes variadas")
    print("           - toxicity_penalty      # Evitar ódio/violencia")
    print("           - misinformation_penalty# Fact-checking")
    print("           + time_well_spent       # Usuário se sente bem depois")

    print()
    print("  Observação histórica:")
    print("  YouTube 2016: otimizou apenas 'watch time'")
    print("  → Recomendou conteúdo extremo que 'prendia' usuários")
    print("  → Solução: adicionar penalidades éticas à reward function")

    print()
    print("─" * 60)
    print("CENARIOS:")
    print("─" * 60)

    cenarios = [
        ("Conteúdo educativo de qualidade",
         InteracaoUsuario(240, 300, True, True, True, False, 4.8, 0.02, 0.01)),
        ("Vídeo viral (engaja, ok)",
         InteracaoUsuario(280, 300, True, True, False, False, 4.2, 0.10, 0.05)),
        ("Conteúdo tóxico (engaja mas prejudica)",
         InteracaoUsuario(200, 250, True, False, True, False, 2.0, 0.85, 0.10)),
        ("Fake news (compartilhada!)",
         InteracaoUsuario(150, 200, False, True, False, True, 1.5, 0.20, 0.90)),
        ("Conteúdo irrelevante (bounce)",
         InteracaoUsuario(10, 300, False, False, False, False, 2.0, 0.05, 0.02)),
    ]

    for desc, interacao in cenarios:
        r = calcular_reward_recomendacao(interacao)
        print(f"\n  [{desc}]")
        for k, v in r.items():
            if k != "total":
                print(f"    {k:<25}: {v:>+7.3f}")
        print(f"    {'TOTAL':<25}: {r['total']:>+7.3f}")

    print()
    print("─" * 60)
    print("SIMULACAO SESSAO (10 CONTEUDOS):")
    print("─" * 60)
    print()
    simular_sessao_recomendacao(n_conteudos=10)

    print()
    print("─" * 60)
    print("DESAFIOS ETICOS DO RL EM RECOMENDACAO:")
    print("─" * 60)
    print("  + Personalização genuína melhora experiência")
    print("  + Feedback massivo (bilhões de interações)")
    print("  - Filter bubble: usuário só vê o que já acredita")
    print("  - Addiction by design: maximizar tempo pode prejudicar saúde")
    print("  - Polarização: conteúdo extremo tende a engajar mais")
    print("  - Regulação GDPR/DSA: plataformas precisam auditar algoritmos")
    print("  - 'Time Well Spent' vs 'Time on Platform': objetivos opostos!")
