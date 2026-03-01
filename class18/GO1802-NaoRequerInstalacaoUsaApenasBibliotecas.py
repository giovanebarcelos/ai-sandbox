# GO1802-NãoRequerInstalaçãoUsaApenasBibliotecas
# Bellman Optimality para V*
V*(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V*(s')]

# Bellman Optimality para Q*
Q*(s,a) = R(s,a) + γ Σ P(s'|s,a) max_a' Q*(s',a')
