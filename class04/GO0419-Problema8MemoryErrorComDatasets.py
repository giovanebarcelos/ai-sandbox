# GO0419-Problema8MemoryErrorComDatasets
# 1. Carregar apenas colunas necessárias
df = pd.read_csv('huge_file.csv', usecols=['col1', 'col2', 'target'])

# 2. Usar chunking
chunk_size = 10000
chunks = []
for chunk in pd.read_csv('huge_file.csv', chunksize=chunk_size):
    # Processar chunk
    processed = chunk[chunk['value'] > 0]
    chunks.append(processed)
df = pd.concat(chunks)

# 3. Otimizar tipos de dados
df['int_col'] = df['int_col'].astype('int32')  # Em vez de int64
df['category_col'] = df['category_col'].astype('category')

# 4. Usar Dask para datasets muito grandes
import dask.dataframe as dd
ddf = dd.read_csv('huge_file.csv')
result = ddf.groupby('category').mean().compute()
