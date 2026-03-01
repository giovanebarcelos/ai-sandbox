# GO1329-HyperparametersCustomizadosPar
# Hyperparameters customizados para seu domínio


if __name__ == "__main__":
    model.train(
        data='data.yaml',

        # SPATIAL AUGMENTATIONS
        degrees=15.0,        # Rotação ±15° (padrão: 0.0)
        translate=0.2,       # Translação 20% (padrão: 0.1)
        scale=0.9,           # Escala 0.5-1.5x (padrão: 0.5)
        shear=5.0,           # Shear ±5° (padrão: 0.0)
        perspective=0.001,   # Perspectiva (padrão: 0.0)
        flipud=0.5,          # Flip vertical 50% (útil para aerial view)
        fliplr=0.5,          # Flip horizontal 50%

        # COLOR AUGMENTATIONS
        hsv_h=0.02,          # Hue shift ±2% (padrão: 0.015)
        hsv_s=0.7,           # Saturation 0.3-1.7x (padrão: 0.7)
        hsv_v=0.4,           # Value 0.6-1.4x (padrão: 0.4)

        # ADVANCED AUGMENTATIONS
        mosaic=1.0,          # Mosaic augmentation (4 imagens em 1)
        mixup=0.15,          # Mixup augmentation (blend 2 imagens)
        copy_paste=0.3,      # Copy-paste augmentation (YOLOv8 específico)

        # REGULARIZATION
        dropout=0.0,         # Dropout (geralmente não usado em detecção)
        label_smoothing=0.1  # Label smoothing (0.0 = off, 0.1 = recomendado)
    )
