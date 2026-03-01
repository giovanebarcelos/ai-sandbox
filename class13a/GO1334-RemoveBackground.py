# GO1334-RemoveBackground
def remove_background(image_path, model):
    results = model.predict(image_path)

    img = cv2.imread(image_path)

    # Criar máscara combinada de todos os objetos
    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)

    for mask in results[0].masks:
        mask_array = mask.data[0].cpu().numpy()
        mask_resized = cv2.resize(mask_array, (img.shape[1], img.shape[0]))
        combined_mask[mask_resized > 0.5] = 255

    # Aplicar máscara
    img_no_bg = cv2.bitwise_and(img, img, mask=combined_mask)

    # Background transparente (se PNG)
    img_rgba = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2BGRA)
    img_rgba[:, :, 3] = combined_mask

    return img_rgba


if __name__ == '__main__':
    print("=== Remove Background com YOLO (demonstração conceitual) ===")
    print()
    print("  Este código requer:")
    print("    pip install ultralytics opencv-python")
    print()
    print("  Uso típico no Google Colab:")
    print("    from ultralytics import YOLO")
    print("    import cv2, numpy as np")
    print()
    print("    model = YOLO('yolov8n-seg.pt')")
    print("    img_rgba = remove_background('foto.jpg', model)")
    print("    cv2.imwrite('result.png', img_rgba)")
    print()

    # Demo com imagem sintética (sem YOLO real)
    try:
        import cv2
        import numpy as np

        # Simular resultado com máscara manual
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[25:75, 25:75] = [0, 120, 255]  # Objeto azul no centro

        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255  # Máscara do objeto

        img_no_bg = cv2.bitwise_and(img, img, mask=mask)
        img_rgba = cv2.cvtColor(img_no_bg, cv2.COLOR_BGR2BGRA)
        img_rgba[:, :, 3] = mask

        cv2.imwrite('remove_bg_demo.png', img_rgba)
        print("  Demo salvo em remove_bg_demo.png (objeto com fundo transparente)")
    except ImportError:
        print("  opencv-python não instalado. Execute: pip install opencv-python")
