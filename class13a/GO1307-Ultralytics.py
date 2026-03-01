# GO1307-Ultralytics
from ultralytics import YOLO

# Carregar modelo pré-treinado (transfer learning)


if __name__ == "__main__":
    model = YOLO('yolov8m.pt')

    # Fine-tuning
    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='epi_detector',

        # Hiperparâmetros
        lr0=0.01,           # Learning rate inicial
        optimizer='Adam',
        patience=50,        # Early stopping

        # Augmentation (já incluso por padrão)
        # hsv_h=0.015, hsv_s=0.7, hsv_v=0.4
        # degrees=0.0, translate=0.1, scale=0.5
        # flipud=0.0, fliplr=0.5
    )

    # Modelo treinado salvo em: runs/detect/epi_detector/weights/best.pt
