# GO1819-Exercício2DqnNoCartpole
   best_action = np.argmax(Q_network.predict(s'))
   target = r + gamma * Q_target.predict(s')[best_action]
