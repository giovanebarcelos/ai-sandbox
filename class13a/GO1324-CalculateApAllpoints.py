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


if __name__ == '__main__':
    print("=== Average Precision (todos os pontos) ===")

    # Curva P-R de exemplo
    recalls    = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    precisions = [0.9, 0.85, 0.80, 0.70, 0.60, 0.40, 0.20]

    ap = calculate_ap_allpoints(precisions, recalls)
    print(f"  AP (área sob curva): {ap:.4f}")

    # Caso perfeito
    recalls_perfeito    = [0.0, 0.5, 1.0]
    precisions_perfeito = [1.0, 1.0, 1.0]
    ap_perfeito = calculate_ap_allpoints(precisions_perfeito, recalls_perfeito)
    print(f"  AP perfeito (esperado=1.0): {ap_perfeito:.4f}")
