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
