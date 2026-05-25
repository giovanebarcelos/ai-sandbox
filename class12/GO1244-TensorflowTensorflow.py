# GO1244-TensorflowTensorflow
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np

# ========================================
# 1. CARREGAR MODELO PRÉ-TREINADO
# ========================================
model = ResNet50(weights='imagenet')

print(f"Total parâmetros: {model.count_params():,}")
# Output: 25,636,712

# ========================================
# 2. FAZER PREDIÇÃO EM IMAGEM
# ========================================
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Predição
preds = model.predict(x)

# Decodificar top-3
print('Predições:')
for i, (imagenet_id, label, score) in enumerate(decode_predictions(preds, top=3)[0]):
    print(f"{i+1}. {label}: {score*100:.2f}%")

# Output exemplo:
# 1. African_elephant: 92.45%
# 2. Indian_elephant: 7.23%
# 3. tusker: 0.18%

    # ─── VISUALIZAÇÃO: TOP-10 PREDIÇÕES ───
    import matplotlib
    import matplotlib.pyplot as plt

    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
    except NameError:
        pass

    top10 = decode_predictions(preds, top=10)[0]
    labels_top10 = [item[1] for item in top10]
    scores_top10 = [item[2] * 100 for item in top10]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#e15759' if i == 0 else '#4e79a7' for i in range(10)]
    bars = ax.barh(range(10), scores_top10, color=colors, edgecolor='black')
    ax.set_yticks(range(10))
    ax.set_yticklabels(labels_top10, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Probabilidade (%)')
    ax.set_title('ResNet50 (ImageNet) — Top-10 Predições', fontsize=12)
    for bar, score in zip(bars, scores_top10):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{score:.2f}%', va='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
                      input_shape=(224, 224, 3))

# Congelar primeiros 140 layers (keep features gerais)
for layer in base_model.layers[:140]:
    layer.trainable = False

# Adicionar camadas customizadas
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Criar modelo final
model = Model(inputs=base_model.input, outputs=predictions)

# Compilar com learning rate baixo (fine-tuning)
model.compile(
    optimizer=Adam(lr=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Treinar só nas camadas novas
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)
