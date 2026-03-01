# GO1321-Predictions

if __name__ == "__main__":
    predictions = [
        {'class': 'car', 'conf': 0.95, 'bbox': [...], 'iou_with_gt': 0.82},
        {'class': 'car', 'conf': 0.88, 'bbox': [...], 'iou_with_gt': 0.91},
        {'class': 'car', 'conf': 0.76, 'bbox': [...], 'iou_with_gt': 0.45},  # FP
        {'class': 'car', 'conf': 0.65, 'bbox': [...], 'iou_with_gt': 0.78},
    ]

    # Ordenar por confiança (já ordenado acima)
    # GT: 3 carros reais na imagem
