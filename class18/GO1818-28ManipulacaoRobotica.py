# GO1818-28ManipulaçãoRobótica
import numpy as np

class RobotArm:
    def __init__(self):
        self.position = np.array([0, 0, 0])  # XYZ
        self.gripper_open = True

    def step(self, action):
        # action: [dx, dy, dz, gripper_action]
        self.position += action[:3] * 0.01  # Movimenta

        if action[3] > 0.5:  # Fecha garra
            self.gripper_open = False
            if self.is_touching_object():
                reward = 1.0  # Pegou!
                done = True
            else:
                reward = -0.1  # Tentou pegar mas falhou
                done = False
        else:
            reward = -0.01  # Penaliza por não fazer nada
            done = False

        state = self.get_camera_image()
        return state, reward, done

    def is_touching_object(self):
        # Sensor de força na garra
        return np.random.random() < 0.8  # Simplificação

# Treinar com PPO ou SAC (algoritmos para contínuo)


if __name__ == '__main__':
    import numpy as np
    np.random.seed(0)

    print("=== Simulação de Braço Robótico ===")
    arm = RobotArm()

    # Corrigir posição para float (original usa int array)
    arm.position = arm.position.astype(np.float64)

    # Implementar get_camera_image() ausente no código original
    arm.get_camera_image = lambda: arm.position.copy()

    print(f"  Posição inicial: {arm.position}")
    print(f"  Garra aberta: {arm.gripper_open}")
    print()

    total_reward = 0
    for step in range(1, 11):
        # Ação aleatória: movimento [dx, dy, dz] + gripper
        action = np.append(np.random.randn(3) * 0.1, np.random.choice([0.0, 1.0]))
        state, reward, done = arm.step(action)
        total_reward += reward
        print(f"  Passo {step:2d}: pos={arm.position.round(3)}, "
              f"reward={reward:+.2f}, done={done}")
        if done:
            print("  Objeto capturado! Episódio encerrado.")
            break

    print(f"\n  Recompensa total: {total_reward:.2f}")
    print("  (Treinar com PPO/SAC para política ótima)")
