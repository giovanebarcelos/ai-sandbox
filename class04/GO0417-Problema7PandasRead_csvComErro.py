# GO0417-Problema7PandasRead_csvComErro
# Tentar diferentes encodings:
encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

for enc in encodings:
    try:
        df = pd.read_csv('dados.csv', encoding=enc)
        print(f"✅ Sucesso com encoding: {enc}")
        break
    except UnicodeDecodeError:
        print(f"❌ Falhou com: {enc}")

# Ou detectar automaticamente:
import chardet
with open('dados.csv', 'rb') as f:
    result = chardet.detect(f.read(10000))
    encoding = result['encoding']

df = pd.read_csv('dados.csv', encoding=encoding)
