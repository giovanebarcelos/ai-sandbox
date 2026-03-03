# GO0327A-27aMctsEAlphagoRevolução
import numpy as np
import math


if __name__ == "__main__":
    print("🎮 MONTE CARLO TREE SEARCH (MCTS) - SIMPLIFICADO")
    print("=" * 70)

    class MCTSNode:
        """Nó da árvore MCTS"""
        def __init__(self, state, parent=None, action=None):
            self.state = state          # Estado do jogo
            self.parent = parent        # Nó pai
            self.action = action        # Ação que levou a este nó
            self.children = []          # Filhos expandidos
            self.visits = 0             # Número de visitas
            self.wins = 0               # Número de vitórias
            self.untried_actions = state.get_legal_actions()  # Ações não tentadas

        def ucb1(self, c=1.41):  # c = √2 típico
            """Upper Confidence Bound para seleção"""
            if self.visits == 0:
                return float('inf')  # Força exploração de nós não visitados

            exploitation = self.wins / self.visits
            exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
            return exploitation + exploration

        def select_child(self):
            """Seleciona filho com maior UCB1"""
            return max(self.children, key=lambda child: child.ucb1())

        def expand(self):
            """Expande um filho não tentado"""
            action = self.untried_actions.pop()
            next_state = self.state.apply_action(action)
            child = MCTSNode(next_state, parent=self, action=action)
            self.children.append(child)
            return child

        def is_fully_expanded(self):
            return len(self.untried_actions) == 0

        def is_terminal(self):
            return self.state.is_terminal()

        def backpropagate(self, result):
            """Propaga resultado até a raiz"""
            self.visits += 1
            self.wins += result
            if self.parent:
                self.parent.backpropagate(result)

    def mcts_search(root_state, num_simulations=1000):
        """
        Monte Carlo Tree Search

        Args:
            root_state: Estado inicial do jogo
            num_simulations: Número de simulações (tipicamente 1000-100000)

        Returns:
            Melhor ação a tomar
        """
        root = MCTSNode(root_state)

        for i in range(num_simulations):
            node = root

            # 1. SELECTION: Desce na árvore usando UCB1
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.select_child()

            # 2. EXPANSION: Adiciona novo filho
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            # 3. SIMULATION: Joga aleatoriamente até o fim
            state = node.state
            while not state.is_terminal():
                action = np.random.choice(state.get_legal_actions())
                state = state.apply_action(action)

            result = state.get_result()  # 1 para vitória, 0 para derrota

            # 4. BACKPROPAGATION: Propaga resultado
            node.backpropagate(result)

        # Retorna ação com mais visitas (mais robusta que maior win rate)
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    # Exemplo: Jogo da Velha (Tic-Tac-Toe)
    class TicTacToe:
        """Estado do jogo da velha para demonstração"""
        def __init__(self, board=None, player=1):
            self.board = board if board is not None else np.zeros((3, 3))
            self.player = player  # 1 ou -1

        def get_legal_actions(self):
            """Retorna posições vazias"""
            return list(zip(*np.where(self.board == 0)))

        def apply_action(self, action):
            """Aplica jogada e retorna novo estado"""
            new_board = self.board.copy()
            new_board[action] = self.player
            return TicTacToe(new_board, -self.player)

        def is_terminal(self):
            """Verifica se jogo acabou"""
            # Verificar linhas, colunas, diagonais
            for i in range(3):
                if abs(self.board[i].sum()) == 3:  # Linha
                    return True
                if abs(self.board[:, i].sum()) == 3:  # Coluna
                    return True
            if abs(self.board.diagonal().sum()) == 3:  # Diagonal principal
                return True
            if abs(np.fliplr(self.board).diagonal().sum()) == 3:  # Diagonal secundária
                return True
            return len(self.get_legal_actions()) == 0  # Empate

        def get_result(self):
            """Retorna 1 se jogador atual venceu, 0 caso contrário"""
            # Simplificado: retorna 1 se X venceu, 0 se O venceu, 0.5 empate
            for i in range(3):
                if self.board[i].sum() == 3 or self.board[:, i].sum() == 3:
                    return 1
                if self.board[i].sum() == -3 or self.board[:, i].sum() == -3:
                    return 0
            if self.board.diagonal().sum() == 3 or np.fliplr(self.board).diagonal().sum() == 3:
                return 1
            if self.board.diagonal().sum() == -3 or np.fliplr(self.board).diagonal().sum() == -3:
                return 0
            return 0.5  # Empate

    print("\n🎯 Exemplo: MCTS jogando Jogo da Velha\n")

    # Estado inicial
    state = TicTacToe()

    print("Tabuleiro inicial:")
    print(state.board)
    print()

    # MCTS encontra melhor jogada
    print("🤖 MCTS pensando... (1000 simulações)")
    best_action = mcts_search(state, num_simulations=1000)
    print(f"✅ Melhor jogada: {best_action}\n")

    print("💡 MCTS explorou:")
    print("   • ~1000 simulações")
    print("   • Balanceou exploitation (melhores jogadas) e exploration (jogadas inexploradas)")
    print("   • Não precisou de heurística!")
    print("   • Aprendeu jogando")

    print("\n🏆 COMPARAÇÃO:")
    print("   Minimax (força bruta): ~500K estados para Go")
    print("   MCTS: ~10K simulações, resultado similar")
    print("   AlphaGo: MCTS + Deep Learning = melhor do mundo!")
