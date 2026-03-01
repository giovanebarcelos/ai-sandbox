# GO1302-DescongelarÚltimasCamadas
# Descongelar últimas camadas
base_model.trainable = True

# Congelar apenas primeiras camadas
for layer in base_model.layers[:100]:
    layer.trainable = False

# Treinar com learning rate BAIXO
model.compile(
    optimizer=Adam(lr=1e-5),  # LR muito menor!
    loss='categorical_crossentropy'
)
