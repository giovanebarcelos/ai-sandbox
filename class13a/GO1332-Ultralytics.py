# GO1332-Ultralytics
from ultralytics import YOLO

# Carregar modelo de segmentação pré-treinado


if __name__ == "__main__":
    model = YOLO('yolov8m-seg.pt')

    # Fine-tuning
    results = model.train(
        data='data.yaml',  # Mesmo formato que detecção
        task='segment',    # Importante: especificar task
        epochs=100,
        imgsz=640,
        batch=8,           # Segmentação usa mais memória
        name='custom_segmentation',

        # Hiperparâmetros
        lr0=0.01,
        optimizer='Adam',

        # Augmentations (mesmas que detecção)
        degrees=10,
        mosaic=1.0,
        copy_paste=0.3     # Especialmente útil para segmentação!
    )

    # Validar
    metrics = model.val()

    print(f"mAP box: {metrics.box.map}")     # mAP para bounding boxes
    print(f"mAP mask: {metrics.seg.map}")    # mAP para máscaras
