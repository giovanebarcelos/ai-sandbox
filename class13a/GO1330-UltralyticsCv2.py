# GO1330-UltralyticsCv2
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('yolov8m-seg.pt')

# Inferência
results = model.predict('image.jpg', conf=0.5)

# Acessar máscaras
for result in results:
    # Bounding boxes (como antes)
    boxes = result.boxes

    # MÁSCARAS DE SEGMENTAÇÃO
    masks = result.masks  # Novo!

    if masks is not None:
        for i, (box, mask) in enumerate(zip(boxes, masks)):
            # Classe e confiança
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]

            # Máscara binária (H x W)
            mask_array = mask.data[0].cpu().numpy()  # 0-1 values

            # Coordenadas da bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            print(f"{class_name} ({conf:.2f}): mask shape {mask_array.shape}")

            # Aplicar máscara colorida na imagem original
            colored_mask = np.zeros_like(result.orig_img)
            color = np.random.randint(0, 255, 3).tolist()

            # Resize mask para tamanho original
            mask_resized = cv2.resize(mask_array, 
                                     (result.orig_img.shape[1], 
                                      result.orig_img.shape[0]))

            # Aplicar cor onde mask > 0.5
            colored_mask[mask_resized > 0.5] = color

            # Blend com imagem original
            alpha = 0.5
            result.orig_img = cv2.addWeighted(result.orig_img, 1, 
                                             colored_mask, alpha, 0)

# Visualizar
cv2.imshow('Segmentation', result.orig_img)
cv2.waitKey(0)
