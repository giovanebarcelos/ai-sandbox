# GO1637-PesosOriginais
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def aplicar_lora(W, A, B, x):
    original = x @ W.T
    adaptation = (x @ A) @ B
    return original + adaptation

if __name__ == "__main__":
    np.random.seed(42)
    d_in, d_out = 768, 768
    ranks = [1, 2, 4, 8, 16, 32, 64]
    print("LoRA - redução de parâmetros")
    for r in ranks:
        params = d_in*r + r*d_out
        pct = params/(d_in*d_out)*100
        print(f"  rank={r}: {params:,} params ({pct:.2f}%)")
    r = 8
    W = np.random.randn(d_out, d_in)*0.01
    A = np.random.randn(d_in, r)*0.01
    B = np.zeros((r, d_out))
    x = np.random.randn(4, d_in)
    out = aplicar_lora(W, A, B, x)
    print(f"output shape: {out.shape}")
    print(f"Redução: {(1-(d_in*r+r*d_out)/(d_in*d_out))*100:.1f}%")
    fig, ax = plt.subplots(figsize=(7,4))
    pcts = [( d_in*r + r*d_out)/(d_in*d_out)*100 for r in ranks]
    ax.plot(ranks, pcts, "o-", color="darkorange", linewidth=2)
    ax.set_xlabel("Rank r"); ax.set_ylabel("% parâmetros")
    ax.set_title("LoRA: Parâmetros vs Rank"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("GO1637_lora_params.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("Salvo: GO1637_lora_params.png")
