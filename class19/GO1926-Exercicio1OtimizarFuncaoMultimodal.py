# GO1926 - Otimizar Funcao Multimodal com Algoritmo Genetico
# f(x,y) = (1-x)^2 * exp(-x^2 - (y+1)^2) - (x - x^3 - y^3) * exp(-x^2 - y^2)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def f(x, y):
    term1 = (1 - x)**2 * np.exp(-x**2 - (y + 1)**2)
    term2 = (x - x**3 - y**3) * np.exp(-x**2 - y**2)
    return term1 - term2

if __name__ == "__main__":
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    print(f"Z: min={Z.min():.3f} max={Z.max():.3f}")
    fig, ax = plt.subplots(figsize=(7, 5))
    cont = ax.contourf(X, Y, Z, levels=30, cmap="viridis")
    plt.colorbar(cont, ax=ax)
    ax.set_title("Funcao Multimodal f(x,y)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig("GO1926_multimodal.png", dpi=100, bbox_inches="tight")
    plt.show()
    print("Salvo: GO1926_multimodal.png")
