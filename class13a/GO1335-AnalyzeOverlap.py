# GO1335-AnalyzeOverlap
def analyze_overlap(results):
    """Detectar se objetos estão sobrepostos"""
    masks = results[0].masks

    if masks is None or len(masks) < 2:
        return

    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            mask1 = masks[i].data[0].cpu().numpy()
            mask2 = masks[j].data[0].cpu().numpy()

            # Interseção
            intersection = np.logical_and(mask1 > 0.5, mask2 > 0.5).sum()

            # Áreas individuais
            area1 = (mask1 > 0.5).sum()
            area2 = (mask2 > 0.5).sum()

            # Porcentagem de sobreposição
            overlap_pct1 = (intersection / area1) * 100
            overlap_pct2 = (intersection / area2) * 100

            cls1 = model.names[int(results[0].boxes[i].cls[0])]
            cls2 = model.names[int(results[0].boxes[j].cls[0])]

            print(f"{cls1} e {cls2}: {overlap_pct1:.1f}% / {overlap_pct2:.1f}% sobrepostos")
