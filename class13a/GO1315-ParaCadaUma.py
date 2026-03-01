# GO1315-ParaCadaUma
# Para cada uma das 3 escalas (80x80, 40x40, 20x20):

Outputs por anchor:
    - Bounding box: [x, y, w, h] (4 valores)
    - Objectness score: [0-1] (1 valor)
    - Class probabilities: [p1, p2, ..., p80] (80 valores para COCO)

Total: 80x80x3 + 40x40x3 + 20x20x3 = 25,200 predições!
