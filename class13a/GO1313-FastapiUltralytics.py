# GO1313-FastapiUltralytics
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np


if __name__ == "__main__":
    app = FastAPI()
    model = YOLO('best.pt')

    @app.post("/detect")
    async def detect(file: UploadFile = File(...)):
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model.predict(img)

        detections = []
        for box in results[0].boxes:
            detections.append({
                'class': model.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'bbox': box.xyxy[0].tolist()
            })

        return {'detections': detections}
