# GO1331-CocoToYoloSeg
import json
import numpy as np
from PIL import Image

def coco_to_yolo_seg(coco_json, output_dir):
    with open(coco_json) as f:
        data = json.load(f)

    # Criar mapping image_id → filename
    images = {img['id']: img for img in data['images']}

    # Processar cada anotação
    for ann in data['annotations']:
        img_id = ann['image_id']
        img_info = images[img_id]
        img_w, img_h = img_info['width'], img_info['height']

        # Segmentation pode ser lista de polígonos
        segmentation = ann['segmentation'][0]  # Pegar primeiro polígono

        # Normalizar coordenadas
        points = []
        for i in range(0, len(segmentation), 2):
            x = segmentation[i] / img_w
            y = segmentation[i+1] / img_h
            points.extend([x, y])

        # Salvar em formato YOLO
        category_id = ann['category_id']
        filename = img_info['file_name'].replace('.jpg', '.txt')

        with open(f"{output_dir}/{filename}", 'a') as f:
            f.write(f"{category_id} {' '.join(map(str, points))}\n")
