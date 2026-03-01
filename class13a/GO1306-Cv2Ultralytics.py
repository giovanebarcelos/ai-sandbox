# GO1306-Cv2Ultralytics
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Modelo leve para tempo real

# Abrir webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferência
    results = model.predict(frame, conf=0.5, verbose=False)

    # Plotar resultados
    annotated_frame = results[0].plot()

    # Mostrar
    cv2.imshow('YOLOv8 Webcam', annotated_frame)

    # Sair com 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
