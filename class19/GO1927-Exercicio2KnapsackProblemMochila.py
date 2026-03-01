# GO1927-Exercício2KnapsackProblemMochila
# Dados exemplo (10 itens)
values = [10, 40, 30, 50, 35, 40, 30, 50, 45, 10]
weights = [2, 5, 10, 5, 4, 3, 1, 7, 6, 18]
capacity = 25

# Cromossomo: [1, 0, 1, 0, 1, ...] (1=incluir, 0=excluir)
# Fitness: sum(v_i * gene_i) se sum(w_i * gene_i) ≤ C, senão penalizar
