# GO1304-UltralyticsCv2
from ultralytics import YOLO
import cv2
from PIL import Image

# Carregar modelo pré-treinado (COCO - 80 classes)
model = YOLO('yolov8m.pt')

# OPÇÃO 1: Inferência simples
results = model('path/to/image.jpg')

# Visualizar resultado
results[0].show()

# OPÇÃO 2: Inferência com controle
results = model.predict(
    source='image.jpg',
    conf=0.5,        # Confiança mínima
    iou=0.7,         # IoU threshold para NMS
    show=True,       # Mostrar resultado
    save=True        # Salvar resultado
)

# ACESSAR DETECÇÕES:
for result in results:
    boxes = result.boxes  # Bounding boxes
    for box in boxes:
        # Coordenadas
        x1, y1, x2, y2 = box.xyxy[0]

        # Confiança e classe
        conf = box.conf[0]
        cls = int(box.cls[0])
        class_name = model.names[cls]

        print(f"{class_name}: {conf:.2f} at ({x1},{y1},{x2},{y2})")
