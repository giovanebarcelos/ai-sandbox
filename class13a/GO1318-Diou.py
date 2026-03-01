# GO1318-Diou
def diou(box1, box2):
    iou = calculate_iou(box1, box2)

    # Distância entre centros
    center1 = get_center(box1)
    center2 = get_center(box2)
    d = euclidean_distance(center1, center2)

    # Diagonal do menor retângulo envolvente
    c = diagonal_length(smallest_enclosing_box(box1, box2))

    # DIoU
    diou = iou - (d**2 / c**2)

    return diou
