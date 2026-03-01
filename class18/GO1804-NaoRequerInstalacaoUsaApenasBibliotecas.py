# GO1804-NãoRequerInstalaçãoUsaApenasBibliotecas
if random() < ε:  # Exploration (explorar novas ações)
    ação = random_action()
else:             # Exploitation (usar melhor conhecimento atual)
    ação = argmax_a Q(s,a)
