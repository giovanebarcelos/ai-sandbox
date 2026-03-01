# GO1324-CalculateApAllpoints
def calculate_ap_allpoints(precisions, recalls):
    # Adicionar pontos sentinela
    precisions = [0] + precisions + [0]
    recalls = [0] + recalls + [1]

    # Tornar precision monotonicamente decrescente
    for i in range(len(precisions)-2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i+1])

    # Calcular área sob a curva
    ap = 0
    for i in range(len(recalls)-1):
        ap += (recalls[i+1] - recalls[i]) * precisions[i+1]

    return ap
