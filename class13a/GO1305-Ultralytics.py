# GO1305-Ultralytics
from ultralytics import YOLO

model = YOLO('yolov8m.pt')

# Processar vídeo
results = model.predict(
    source='video.mp4',
    show=True,           # Mostrar em tempo real
    save=True,           # Salvar vídeo processado
    conf=0.5,
    stream=True          # Processar frame a frame (eficiente)
)

# Processar cada frame
for result in results:
    frame = result.orig_img
    boxes = result.boxes
    # ... processar ...
