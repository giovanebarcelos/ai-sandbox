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


if __name__ == '__main__':
    import json
    import os
    import tempfile

    print("=== Demonstração COCO → YOLO Segmentation ===")

    # Criar JSON COCO mínimo para teste
    coco_data = {
        "images": [
            {"id": 1, "file_name": "foto001.jpg", "width": 640, "height": 480}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 0,
                "segmentation": [[100, 100, 200, 100, 200, 200, 100, 200]]
            }
        ]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        coco_path = os.path.join(tmpdir, "annotations.json")
        out_dir   = os.path.join(tmpdir, "labels")
        os.makedirs(out_dir, exist_ok=True)

        with open(coco_path, 'w') as f:
            json.dump(coco_data, f)

        coco_to_yolo_seg(coco_path, out_dir)

        # Ler resultado
        label_file = os.path.join(out_dir, "foto001.txt")
        with open(label_file) as f:
            conteudo = f.read().strip()
        print(f"  Arquivo YOLO gerado: {conteudo}")
        print("  Formato: <class_id> <x1_norm> <y1_norm> ... <xN_norm> <yN_norm>")
