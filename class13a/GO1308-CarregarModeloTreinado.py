# GO1308-CarregarModeloTreinado
# Carregar modelo treinado


if __name__ == "__main__":
    model = YOLO('runs/detect/epi_detector/weights/best.pt')

    # Inferência
    results = model.predict('new_image.jpg', conf=0.5)
    results[0].show()
