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
