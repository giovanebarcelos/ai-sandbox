# GO1319-Ciou
def ciou(box1, box2):
    iou = calculate_iou(box1, box2)

    # Termo de distância (como DIoU)
    d = distance_term(box1, box2)

    # Consistência de aspect ratio
    v = (4 / π²) * (arctan(w_gt/h_gt) - arctan(w_pred/h_pred))**2

    # Parâmetro de trade-off
    α = v / (1 - iou + v)

    # CIoU
    ciou = iou - d - α * v

    return ciou
