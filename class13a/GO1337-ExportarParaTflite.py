# GO1337-ExportarParaTflite
# Exportar para TFLite com quantização INT8


if __name__ == "__main__":
    model.export(
        format='tflite',
        int8=True,              # Quantização INT8
        data='data.yaml'        # Dataset para calibração
    )

    # RESULTADO:
    # - Tamanho: ~75% menor
    # - Velocidade: ~2-3x mais rápido (em hardware apropriado)
    # - Precisão: -1% a -3% mAP (geralmente aceitável)
