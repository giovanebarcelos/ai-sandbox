# GO1319-Ciou
import math
def ciou(box1, box2):
    iou = calculate_iou(box1, box2)

    # Termo de distância (como DIoU)
    d = distance_term(box1, box2)

    # Consistência de aspect ratio
    v = (4 / (math.pi**2)) * (math.atan(w_gt/h_gt) - math.atan(w_pred/h_pred))**2

    # Parâmetro de trade-off
    alpha = v / (1 - iou + v)

    # CIoU
    ciou = iou - d - alpha * v

    return ciou
