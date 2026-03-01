# GO1311-JáInclusoNo
# Já incluso no YOLO, mas pode personalizar:


if __name__ == "__main__":
    model.train(
        data='data.yaml',
        augment=True,
        hsv_h=0.015,    # Hue augmentation
        hsv_s=0.7,      # Saturation
        hsv_v=0.4,      # Value
        degrees=10.0,   # Rotação
        translate=0.2,  # Translação
        scale=0.5,      # Escala
        fliplr=0.5,     # Flip horizontal
        mosaic=1.0,     # Mosaic augmentation (YOLO específico)
        mixup=0.1       # Mixup augmentation
    )
