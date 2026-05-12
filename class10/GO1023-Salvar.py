# GO1023-Salvar
# Demonstra o formato SavedModel do TensorFlow: salva o modelo como diretório
# com arquivos .pb e variáveis separados, e como recarregá-lo posteriormente.
# Salvar (cria diretório)
model.save('saved_model/')

# Estrutura criada:
# saved_model/
# ├── saved_model.pb
# ├── variables/
# │   ├── variables.data-00000-of-00001
# │   └── variables.index
# └── assets/

# Carregar
model = keras.models.load_model('saved_model/')
