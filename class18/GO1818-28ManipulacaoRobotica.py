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
