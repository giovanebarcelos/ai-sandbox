# GO0107-UsandoModeloPrétreinadoMobilenetv2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Carregar modelo treinado em 1.4M imagens (ImageNet)
modelo = MobileNetV2(weights='imagenet')

# OPÇÃO 1: Usar imagem local
# Salve uma imagem como 'gato.jpg' na mesma pasta do script
img = image.load_img('gato.jpg', target_size=(224, 224))

# OPÇÃO 2: Baixar imagem da internet (requer urllib e PIL)
# from urllib.request import urlretrieve
# from PIL import Image
# url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/640px-Cat03.jpg'
# urlretrieve(url, 'gato.jpg')
# img = image.load_img('gato.jpg', target_size=(224, 224))

# Processar imagem
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predição
preds = modelo.predict(x)
resultado = decode_predictions(preds, top=3)[0]

for i, (_, label, score) in enumerate(resultado):
    print(f"{i+1}. {label}: {score*100:.2f}%")

# Saída:
# 1. tabby_cat: 85.43%
# 2. tiger_cat: 12.31%
# 3. egyptian_cat: 1.89%
