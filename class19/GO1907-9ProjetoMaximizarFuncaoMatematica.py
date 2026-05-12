# GO1907-9ProjetoMaximizarFunçãoMatemática
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

# Função objetivo
def fitness_function(x):
    return x * np.sin(10 * np.pi * x) + 1.0

# Visualizar
x = np.linspace(-1, 2, 1000)
y = fitness_function(x)
plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Função com múltiplos ótimos locais')
plt.grid(True)
plt.show()

# Máximo global aproximadamente em x ≈ 1.85, f(x) ≈ 2.85
