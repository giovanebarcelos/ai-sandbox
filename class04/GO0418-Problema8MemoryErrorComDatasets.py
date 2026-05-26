# GO0418 - Problema 8: MemoryError ao carregar dataset grande
# ERRO COMUM: tentar carregar arquivo enorme de uma vez na RAM
# SOLUCAO: usar chunksize no pandas, ou sklearn partial_fit
#
# Mensagem de erro original:
#   MemoryError: Unable to allocate array...
import io
import pandas as pd
import numpy as np

# Simular um CSV grande em memoria
linhas = ["col1,col2,label"]
for i in range(1000):
    linhas.append(f"{np.random.randn():.4f},{np.random.randn():.4f},{i%3}")
csv_grande = "\n".join(linhas)

# SOLUCAO 1: ler em chunks
print("Lendo em chunks de 200 linhas:")
chunks = []
for chunk in pd.read_csv(io.StringIO(csv_grande), chunksize=200):
    resumo = chunk.describe().loc[['mean']].round(3)
    chunks.append(resumo)
    print(f"  Chunk: {len(chunk)} linhas, media col1={chunk['col1'].mean():.3f}")
print(f"Total de chunks processados: {len(chunks)}")

# SOLUCAO 2: sklearn IncrementalPCA (partial_fit)
from sklearn.decomposition import IncrementalPCA
ipca = IncrementalPCA(n_components=1)
df_full = pd.read_csv(io.StringIO(csv_grande))
X = df_full[['col1','col2']].values
ipca.partial_fit(X[:500])
ipca.partial_fit(X[500:])
X_red = ipca.transform(X)
print(f"\nIncrementalPCA: {X.shape} -> {X_red.shape}")
