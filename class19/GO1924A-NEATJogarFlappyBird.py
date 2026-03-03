# GO1924A-NEATJogarFlappyBird
import neat
import numpy as np

# Simular jogo Flappy Bird (simplificado)
class FlappyBirdGame:
    def __init__(self):
        self.bird_y = 50  # Posição vertical
        self.bird_vel = 0
        self.pipe_x = 100
        self.pipe_gap_y = 50  # Centro do gap
        self.score = 0
        self.alive = True

    def update(self, jump_action):
        """Atualizar física do jogo"""
        # Gravidade
        self.bird_vel += 0.5
        self.bird_y += self.bird_vel

        # Pular
        if jump_action:
            self.bird_vel = -8

        # Mover cano
        self.pipe_x -= 2
        if self.pipe_x < 0:
            self.pipe_x = 100
            self.pipe_gap_y = np.random.randint(30, 70)
            self.score += 1

        # Colisão
        if self.bird_y < 0 or self.bird_y > 100:
            self.alive = False
        if 10 < self.pipe_x < 20:  # Bird passa pelo cano
            if not (self.pipe_gap_y - 15 < self.bird_y < self.pipe_gap_y + 15):
                self.alive = False

        return self.alive

    def get_state(self):
        """Estado observável pela rede neural"""
        return [
            self.bird_y,
            self.bird_vel,
            self.pipe_x,
            self.pipe_gap_y - self.bird_y  # Distância vertical ao gap
        ]

def eval_genome(genome, config):
    """Avaliar fitness de um genoma NEAT"""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    game = FlappyBirdGame()

    frames = 0
    while game.alive and frames < 1000:
        # Rede neural decide: pular ou não
        state = game.get_state()
        output = net.activate(state)
        jump = output[0] > 0.5

        game.update(jump)
        frames += 1

    # Fitness: score * 100 + frames sobrevividos
    fitness = game.score * 100 + frames
    return fitness

# Configurar NEAT (requer config file - exemplo simplificado)
print("🐦 NEAT para Flappy Bird:")
print("  NEAT evolui arquitetura neural + pesos")
print("  Geração 1-10: morre rápido")
print("  Geração 50: passa 5-10 canos")
print("  Geração 100+: joga indefinidamente")
