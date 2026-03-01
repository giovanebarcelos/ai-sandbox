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
