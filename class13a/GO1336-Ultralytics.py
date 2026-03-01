# GO1336-Ultralytics
from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO('best.pt')

    # 1. ONNX (Open Neural Network Exchange) - Cross-platform
    model.export(format='onnx')
    # → best.onnx

    # 2. TensorRT (NVIDIA GPUs) - Máximo performance
    model.export(format='engine', device=0)
    # → best.engine

    # 3. CoreML (Apple iOS/macOS)
    model.export(format='coreml')
    # → best.mlmodel

    # 4. TensorFlow Lite (Android, Raspberry Pi)
    model.export(format='tflite')
    # → best.tflite

    # 5. TensorFlow SavedModel
    model.export(format='saved_model')
    # → best_saved_model/

    # 6. OpenVINO (Intel hardware)
    model.export(format='openvino')
    # → best_openvino_model/
