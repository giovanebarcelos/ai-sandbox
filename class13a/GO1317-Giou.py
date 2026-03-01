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


if __name__ == '__main__':
    print("=== Demonstração GIoU ===")

    # Implementar funções auxiliares necessárias
    def calculate_intersection(b1, b2):
        """b = [x1, y1, x2, y2]"""
        ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    def calculate_union(b1, b2):
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
        return a1 + a2 - calculate_intersection(b1, b2)

    def smallest_enclosing_box(b1, b2):
        w = max(b1[2], b2[2]) - min(b1[0], b2[0])
        h = max(b1[3], b2[3]) - min(b1[1], b2[1])
        return w * h

    # Cenários de teste
    casos = [
        ([0, 0, 4, 4], [0, 0, 4, 4], "idênticas"),
        ([0, 0, 4, 4], [2, 2, 6, 6], "sobrepostas"),
        ([0, 0, 2, 2], [3, 3, 5, 5], "adjacentes sem sobreposição"),
        ([0, 0, 1, 1], [9, 9, 10, 10], "muito distantes"),
    ]
    for box1, box2, desc in casos:
        score = giou(box1, box2)
        print(f"  {desc:35s}: GIoU = {score:+.4f}")
    print("  (GIoU = +1.0 → idênticas | GIoU = -1.0 → muito distantes)")
