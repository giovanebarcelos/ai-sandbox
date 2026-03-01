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
