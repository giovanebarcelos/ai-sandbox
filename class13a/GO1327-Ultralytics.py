# GO1327-Ultralytics
from ultralytics import YOLO

model = YOLO('yolov8m.pt')

# FASE 1: Treinar só a HEAD (10 épocas)
for name, param in model.model.named_parameters():
    if 'model.22' not in name:  # model.22 = head
        param.requires_grad = False

model.train(
    data='data.yaml',
    epochs=10,
    lr0=0.01,
    name='phase1_head_only'
)

# FASE 2: Descongelar últimas camadas do NECK (10 épocas)
for name, param in model.model.named_parameters():
    if any(x in name for x in ['model.18', 'model.19', 'model.20', 'model.21', 'model.22']):
        param.requires_grad = True

model.train(
    data='data.yaml',
    epochs=10,
    lr0=0.001,  # LR menor!
    name='phase2_neck_head',
    resume=True
)

# FASE 3: Descongelar tudo (20 épocas)
for param in model.model.parameters():
    param.requires_grad = True

model.train(
    data='data.yaml',
    epochs=20,
    lr0=0.0001,  # LR muito menor!
    name='phase3_full',
    resume=True
)
