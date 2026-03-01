# GO1322-Codigo
conf_threshold = 0.5
iou_threshold = 0.5

# Percorrer predições em ordem decrescente de confiança:

# Predição 1 (conf=0.95, IoU=0.82)
TP = 1, FP = 0, FN = 2
Precision = 1/1 = 1.00
Recall = 1/3 = 0.33

# Predição 2 (conf=0.88, IoU=0.91)
TP = 2, FP = 0, FN = 1
Precision = 2/2 = 1.00
Recall = 2/3 = 0.67

# Predição 3 (conf=0.76, IoU=0.45) → IoU < 0.5 = FP
TP = 2, FP = 1, FN = 1
Precision = 2/3 = 0.67
Recall = 2/3 = 0.67

# Predição 4 (conf=0.65, IoU=0.78)
TP = 3, FP = 1, FN = 0
Precision = 3/4 = 0.75
Recall = 3/3 = 1.00
