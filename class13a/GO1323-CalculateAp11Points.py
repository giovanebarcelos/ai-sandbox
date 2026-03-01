# GO1323-CalculateAp11Points
def calculate_ap_11points(precisions, recalls):
    ap = 0
    for recall_threshold in [0.0, 0.1, 0.2, ..., 1.0]:  # 11 pontos
        # Pegar máxima precision para recall >= threshold
        max_precision = max([p for r, p in zip(recalls, precisions) 
                            if r >= recall_threshold])
        ap += max_precision

    ap = ap / 11  # Média dos 11 pontos
    return ap


if __name__ == '__main__':
    # Curva Precisão-Recall de exemplo (detector hipotético)
    recalls    = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    precisions = [1.0, 0.9, 0.9, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

    # Nota: a implementação original usa `...` como pseudo-código Python.
    # A versão corrigida usa range(11) / 10 para os 11 limiares.
    def calculate_ap_11points_corrected(precisions, recalls):
        ap = 0
        for t in range(11):  # 0.0, 0.1, ..., 1.0
            recall_threshold = t / 10
            candidates = [p for r, p in zip(recalls, precisions)
                          if r >= recall_threshold]
            if candidates:
                ap += max(candidates)
        return ap / 11

    ap = calculate_ap_11points_corrected(precisions, recalls)
    print(f"AP (11 pontos): {ap:.4f}")
    print("(Esperado ~0.6–0.7 para esta curva)")
