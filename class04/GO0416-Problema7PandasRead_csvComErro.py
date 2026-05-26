# GO0416 - Problema 7: UnicodeDecodeError ao ler CSV com pandas
# ERRO COMUM: arquivo CSV salvo em latin-1 ou cp1252, nao UTF-8
# SOLUCAO: especificar encoding correto ou usar errors='replace'
#
# Mensagem de erro original:
#   UnicodeDecodeError: 'utf-8' codec can't decode byte...
import io
import pandas as pd

# Simular CSV com encoding latin-1
csv_latin = "nome,valor\nJoao,10\nMaria,20\nAndre,30\n"
csv_bytes_latin = csv_latin.encode('latin-1')

print("Lendo CSV com encoding correto (latin-1):")
df = pd.read_csv(io.BytesIO(csv_bytes_latin), encoding='latin-1')
print(df)

print("\nAlternativa: encoding='cp1252' (Windows):")
df2 = pd.read_csv(io.BytesIO(csv_bytes_latin), encoding='cp1252')
print(df2)

print("\nAlternativa robusta: errors='replace':")
df3 = pd.read_csv(io.BytesIO(csv_bytes_latin), encoding='utf-8', errors='replace')
print(df3)
