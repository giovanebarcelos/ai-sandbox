# GO1340-UltralyticsGlob
from ultralytics import YOLO
import glob

model = YOLO('yolov8m.pt')

# Processar múltiplas imagens em batch
image_paths = glob.glob('dataset/*.jpg')

# OPÇÃO 1: Batch inference (mais eficiente)
results = model.predict(
    source=image_paths,
    batch=16,           # Processar 16 imagens por vez
    stream=True,        # Generator (eficiente em memória)
    device=0            # GPU
)

for result in results:
    # Processar resultado
    result.save(f'output/{result.path}')

# SPEEDUP vs processar 1 por 1: ~3-4x!
