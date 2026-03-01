# GO1816-NãoRequerInstalaçãoUsaApenasBibliotecas
# Resetar ambiente


if __name__ == "__main__":
    observation = env.reset()  # Retorna estado inicial
    # observation: array([x, x_dot, theta, theta_dot]) para CartPole

    # Espaços de observação e ação
    print(env.observation_space)  # Box(4,) - 4 valores contínuos
    print(env.action_space)       # Discrete(2) - 2 ações discretas

    # Step: executar ação
    observation, reward, terminated, truncated, info = env.step(action)
    # observation: próximo estado
    # reward: recompensa recebida
    # terminated: episódio terminou (objetivo atingido/falhou)
    # truncated: episódio cortado (limite de steps)
    # info: dict com informações extras

    done = terminated or truncated

    # Renderizar (visualizar)
    env.render()  # Mostra interface gráfica

    # Fechar ambiente
    env.close()
