# GO1335-AnalyzeOverlap
def analyze_overlap(results):
    """Detectar se objetos estão sobrepostos"""
    masks = results[0].masks

    if masks is None or len(masks) < 2:
        return

    for i in range(len(masks)):
        for j in range(i+1, len(masks)):
            mask1 = masks[i].data[0].cpu().numpy()
            mask2 = masks[j].data[0].cpu().numpy()

            # Interseção
            intersection = np.logical_and(mask1 > 0.5, mask2 > 0.5).sum()

            # Áreas individuais
            area1 = (mask1 > 0.5).sum()
            area2 = (mask2 > 0.5).sum()

            # Porcentagem de sobreposição
            overlap_pct1 = (intersection / area1) * 100
            overlap_pct2 = (intersection / area2) * 100

            cls1 = model.names[int(results[0].boxes[i].cls[0])]
            cls2 = model.names[int(results[0].boxes[j].cls[0])]

            print(f"{cls1} e {cls2}: {overlap_pct1:.1f}% / {overlap_pct2:.1f}% sobrepostos")


if __name__ == '__main__':
    import numpy as np
    print("=== Demonstração de Análise de Sobreposição ===")
    print()

    # Criar resultado YOLO simulado (sem modelo real)
    class FakeTensor:
        def __init__(self, data):
            self._data = np.array(data, dtype=np.float32)
        def cpu(self):
            return self
        def numpy(self):
            return self._data

    class FakeMask:
        def __init__(self, data):
            self.data = [FakeTensor(data)]

    class FakeBox:
        def __init__(self, cls_id):
            self.cls = [cls_id]

    class FakeResults:
        def __init__(self):
            # Duas máscaras que se sobrepõem no centro
            m1 = np.zeros((100, 100), dtype=np.float32)
            m2 = np.zeros((100, 100), dtype=np.float32)
            m1[10:60, 10:60] = 1.0   # Objeto 1: quadrado 50x50
            m2[40:90, 40:90] = 1.0   # Objeto 2: quadrado 50x50 (sobrepõe 20x20)
            self.masks = [FakeMask(m1), FakeMask(m2)]
            self.boxes = [FakeBox(0), FakeBox(1)]

    # Substituição temporária de `model.names`
    import builtins
    class _FakeModel:
        names = {0: "Cachorro", 1: "Gato"}
    model = _FakeModel()  # necessário para analyze_overlap usar model.names

    results = [FakeResults()]
    analyze_overlap(results)
