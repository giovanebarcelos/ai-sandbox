# GO1801-NãoRequerInstalaçãoUsaApenasBibliotecas
# State Value Function
V^π(s) = recompensa esperada partindo de s seguindo π
       = E[R_t + γR_{t+1} + γ²R_{t+2} + ... | s_t=s, π]

# Action Value Function (Q-Function)
Q^π(s,a) = recompensa esperada tomando a em s e seguindo π
         = E[R_t + γR_{t+1} + γ²R_{t+2} + ... | s_t=s, a_t=a, π]
