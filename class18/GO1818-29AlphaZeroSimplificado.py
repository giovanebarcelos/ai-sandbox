# GO1818-29AlphaZeroSimplificado
class AlphaZeroNode:
    def __init__(self, state):
        self.state = state
        self.children = []
        self.visits = 0
        self.value = 0
        self.prior = 0  # Probabilidade neural network

    def select_child(self):
        # UCB (Upper Confidence Bound)
        scores = [
            child.value / (child.visits + 1) +  # Exploitation
            c * child.prior * np.sqrt(self.visits) / (child.visits + 1)  # Exploration
            for child in self.children
        ]
        return self.children[np.argmax(scores)]

    def backpropagate(self, value):
        self.visits += 1
        self.value += value
        if self.parent:
            self.parent.backpropagate(-value)  # Adversarial


if __name__ == '__main__':
    import numpy as np
    np.random.seed(42)

    print("=== Demonstração AlphaZero Node ===")

    # Patch: adicionar atributo 'parent' que o código original referencia mas não inicializa
    original_init = AlphaZeroNode.__init__
    def patched_init(self, state):
        original_init(self, state)
        self.parent = None  # necessário para backpropagate()
    AlphaZeroNode.__init__ = patched_init

    # Patch: 'c' no select_child é a constante UCB, não definida no código original
    # (bug de nomenclatura: variável de iteração 'child' choca com 'c' implícito)
    C_PUCT = 1.0
    def fixed_select_child(self):
        scores = [
            child.value / (child.visits + 1) +
            C_PUCT * child.prior * np.sqrt(self.visits) / (child.visits + 1)
            for child in self.children
        ]
        return self.children[np.argmax(scores)]
    AlphaZeroNode.select_child = fixed_select_child

    # Criar árvore MCTS simples de 3 níveis
    root = AlphaZeroNode(state="raiz")
    root.parent = None

    # Adicionar filhos com priors
    for i in range(3):
        child = AlphaZeroNode(state=f"filho_{i}")
        child.parent = root
        child.prior = np.random.dirichlet([1, 1, 1])[i]
        child.visits = np.random.randint(1, 20)
        child.value = np.random.uniform(-1, 1)
        root.children.append(child)

    root.visits = sum(c.visits for c in root.children)

    print(f"  Raiz: visits={root.visits}")
    for c in root.children:
        print(f"    {c.state}: visits={c.visits}, value={c.value:.3f}, prior={c.prior:.3f}")

    melhor = root.select_child()
    print(f"\n  Filho selecionado (UCB): {melhor.state}")

    # Backpropagate
    root.backpropagate(0.7)
    print(f"  Após backpropagate(0.7): raiz visits={root.visits}, value={root.value:.3f}")
