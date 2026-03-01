# GO1317-Giou
def giou(box1, box2):
    # IoU normal
    intersection = calculate_intersection(box1, box2)
    union = calculate_union(box1, box2)
    iou = intersection / union

    # Área do menor retângulo envolvente
    C = smallest_enclosing_box(box1, box2)

    # GIoU
    giou = iou - (C - union) / C

    return giou

# Valores: -1 ≤ GIoU ≤ 1
# GIoU = 1: boxes idênticas
# GIoU = -1: boxes muito distantes
