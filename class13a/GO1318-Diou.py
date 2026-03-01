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


if __name__ == '__main__':
    import math
    print("=== Demonstração DIoU ===")

    # Implementar funções auxiliares necessárias
    def calculate_iou(b1, b2):
        ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
        ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        union = a1 + a2 - inter
        return inter / union if union > 0 else 0.0

    def get_center(b):
        return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)

    def euclidean_distance(p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def smallest_enclosing_box(b1, b2):
        return [min(b1[0],b2[0]), min(b1[1],b2[1]),
                max(b1[2],b2[2]), max(b1[3],b2[3])]

    def diagonal_length(b):
        return math.sqrt((b[2]-b[0])**2 + (b[3]-b[1])**2)

    casos = [
        ([0, 0, 4, 4], [0, 0, 4, 4], "idênticas"),
        ([0, 0, 4, 4], [2, 2, 6, 6], "sobrepostas"),
        ([0, 0, 2, 2], [3, 3, 5, 5], "sem sobreposição, próximas"),
        ([0, 0, 1, 1], [9, 9, 10, 10], "muito distantes"),
    ]
    for box1, box2, desc in casos:
        score = diou(box1, box2)
        print(f"  {desc:35s}: DIoU = {score:+.4f}")
