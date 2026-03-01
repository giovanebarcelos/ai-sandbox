# GO1907-9ProjetoMaximizarFunçãoMatemática
import numpy as np
import matplotlib.pyplot as plt

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
