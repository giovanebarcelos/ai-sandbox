# GO1328-Torch
# Ultralytics não suporta diretamente, mas pode fazer custom training loop:

import torch.optim as optim

# Separar parâmetros por grupo
backbone_params = []
neck_params = []
head_params = []

for name, param in model.named_parameters():
    if 'model.0' <= name <= 'model.9':  # backbone
        backbone_params.append(param)
    elif 'model.10' <= name <= 'model.21':  # neck
        neck_params.append(param)
    else:  # head
        head_params.append(param)

# Criar optimizer com LRs diferentes
optimizer = optim.Adam([
    {'params': backbone_params, 'lr': 1e-5},  # LR muito baixo
    {'params': neck_params, 'lr': 1e-4},      # LR médio
    {'params': head_params, 'lr': 1e-3}       # LR alto
])
