# GO1341-Ultralytics
from ultralytics import YOLO

model = YOLO('yolov8m.pt')

# Tracking com ByteTrack
results = model.track(
    source='video.mp4',
    tracker='bytetrack.yaml',  # ou 'botsort.yaml'
    conf=0.5,
    iou=0.7,
    show=True,
    save=True
)

# Acessar IDs dos objetos
for result in results:
    if result.boxes.id is not None:
        track_ids = result.boxes.id.cpu().numpy().astype(int)
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for track_id, box, cls in zip(track_ids, boxes, classes):
            x1, y1, x2, y2 = box
            class_name = model.names[cls]

            print(f"ID {track_id}: {class_name} at ({x1},{y1},{x2},{y2})")
