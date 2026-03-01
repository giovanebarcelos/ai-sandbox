# GO0910-Adam
class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8
        self.m = None  # 1º momento
        self.v = None  # 2º momento
        self.t = 0     # timestep

    def update(self, w, grad):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)

        self.t += 1

        # Atualizar momentos
        self.m = self.beta1*self.m + (1-self.beta1)*grad
        self.v = self.beta2*self.v + (1-self.beta2)*grad**2

        # Correção de bias
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Atualizar peso
        w -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return w


if __name__ == '__main__':
    import numpy as np
    np.random.seed(0)

    print("=== Demonstração do Otimizador Adam ===")
    # Minimizar f(w) = w^2 (gradiente = 2w)
    w = np.array([5.0, -3.0, 2.0])
    opt = AdamOptimizer(lr=0.1)

    print(f"w inicial: {w}")
    for step in range(1, 21):
        grad = 2 * w        # f(w) = w² → grad = 2w
        w = opt.update(w, grad)
        if step % 5 == 0:
            print(f"  Passo {step:2d}: w = {w}  ||w|| = {np.linalg.norm(w):.6f}")

    print(f"w final (esperado ≈ 0): {w}")
