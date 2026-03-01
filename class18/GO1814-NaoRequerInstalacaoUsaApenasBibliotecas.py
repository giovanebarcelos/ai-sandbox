# GO1814-NãoRequerInstalaçãoUsaApenasBibliotecas
# PPO clipping objective


if __name__ == "__main__":
    ratio = π_θ_new(a|s) / π_θ_old(a|s)
    clipped_ratio = clip(ratio, 1-ε, 1+ε)  # ε=0.2 típico
    L = min(ratio * A(s,a), clipped_ratio * A(s,a))
