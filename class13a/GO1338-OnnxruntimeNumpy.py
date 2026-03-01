# GO1338-OnnxruntimeNumpy
import onnxruntime as ort
import numpy as np
import cv2

# Carregar modelo ONNX


if __name__ == "__main__":
    session = ort.InferenceSession('best.onnx')

    # Pré-processar imagem
    img = cv2.imread('image.jpg')
    img_resized = cv2.resize(img, (640, 640))
    img_normalized = img_resized / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1))
    img_batch = np.expand_dims(img_transposed, axis=0).astype(np.float32)

    # Inferência
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img_batch})

    # Processar outputs
    predictions = outputs[0]
    # ... pós-processamento (NMS, etc.)
