# GO1333-CalcularÁreaOcupada
# Calcular área ocupada por cada classe
for result in results:
    masks = result.masks
    for mask, box in zip(masks, result.boxes):
        cls = int(box.cls[0])
        mask_array = mask.data[0].cpu().numpy()

        # Número de pixels
        num_pixels = (mask_array > 0.5).sum()

        # Área em metros² (se souber escala)
        scale = 0.01  # 1 pixel = 0.01 m²
        area_m2 = num_pixels * scale

        print(f"{model.names[cls]}: {area_m2:.2f} m²")
