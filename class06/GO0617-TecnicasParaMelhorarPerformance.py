# GO0617-TécnicasParaMelhorarPerformance
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer

# Criar dataset de exemplo


if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'area': np.random.uniform(50, 300, n),
        'preco': np.random.uniform(200, 800, n),
        'n_quartos': np.random.randint(1, 6, n),
        'n_comodos': np.random.randint(3, 10, n),
        'idade_imovel': np.random.uniform(0, 50, n),
        'bairro': np.random.choice(['Centro', 'Norte', 'Sul', 'Leste'], n),
        'tipo': np.random.choice(['Casa', 'Apartamento'], n),
        'data_venda': pd.date_range('2020-01-01', periods=n, freq='D')
    })

    # 1. TRANSFORMAÇÕES
    df['log_area'] = np.log1p(df['area'])  # log(1+x) evita log(0)
    df['sqrt_idade'] = np.sqrt(df['idade_imovel'])

    # Box-Cox (requer valores > 0)
    pt = PowerTransformer(method='box-cox')
    df['area_transformed'] = pt.fit_transform(df[['area']])

    # 2. INTERAÇÕES
    df['preco_por_m2'] = df['preco'] / df['area']
    df['area_x_quartos'] = df['area'] * df['n_quartos']
    df['densidade'] = df['n_comodos'] / df['area']

    # 3. BINNING
    df['faixa_area'] = pd.cut(df['area'], 
                               bins=[0, 50, 100, 200, 500],
                               labels=['Pequeno', 'Médio', 'Grande', 'Enorme'])

    df['faixa_preco'] = pd.qcut(df['preco'], q=4, 
                                 labels=['Baixo', 'Médio', 'Alto', 'Premium'])

    # 4. ENCODING
    # One-Hot
    df_encoded = pd.get_dummies(df, columns=['bairro', 'tipo'], 
                                  drop_first=True)  # Evita multicolinearidade

    # Target Encoding (manual - sem dependência externa)
    # Média do target por categoria
    bairro_means = df.groupby('bairro')['preco'].mean()
    df['bairro_encoded'] = df['bairro'].map(bairro_means)

    # 5. FEATURES TEMPORAIS
    df['ano'] = df['data_venda'].dt.year
    df['mes'] = df['data_venda'].dt.month
    df['dia_semana'] = df['data_venda'].dt.dayofweek
    df['trimestre'] = df['data_venda'].dt.quarter
    df['é_fim_de_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
    df['dias_desde_2020'] = (df['data_venda'] - pd.Timestamp('2020-01-01')).dt.days

    print("="*60)
    print("FEATURE ENGINEERING APLICADO")
    print("="*60)
    print(f"\nFeatures originais: {['area', 'preco', 'n_quartos', 'n_comodos', 'idade_imovel']}")
    print(f"Total de features após engineering: {df.shape[1]}")
    print(f"\nNovas features criadas:")
    print(f"  • Transformações: log_area, sqrt_idade, area_transformed")
    print(f"  • Interações: preco_por_m2, area_x_quartos, densidade")
    print(f"  • Binning: faixa_area, faixa_preco")
    print(f"  • Encoding: bairro_encoded")
    print(f"  • Temporais: ano, mes, dia_semana, trimestre, é_fim_de_semana, dias_desde_2020")
    print(f"\nPrimeiras linhas:")
    print(df[['area', 'log_area', 'preco_por_m2', 'faixa_area', 'mes']].head())
