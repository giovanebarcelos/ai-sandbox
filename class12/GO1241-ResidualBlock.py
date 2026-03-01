# GO1241-ResidualBlock
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    Bloco residual básico (usado em ResNet-18/34)
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Caminho principal (F(x))
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity)
        self.shortcut = nn.Sequential()

        # Se dimensões não batem, usar projeção 1×1
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # Salvar input para skip connection
        identity = x

        # Caminho principal: F(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Skip connection: F(x) + x
        identity = self.shortcut(identity)  # Ajustar dimensões se necessário
        out += identity  # ⭐ ELEMENTO-WISE ADD!

        # ReLU DEPOIS da adição
        out = self.relu(out)

        return out
