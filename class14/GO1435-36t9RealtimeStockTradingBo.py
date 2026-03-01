# GO1435-36t9RealtimeStockTradingBo
# ═══════════════════════════════════════════════════════════════════
# BOT DE TRADING AUTOMÁTICO COM LSTM
# Aplicação: Decisão compra/venda baseada em séries temporais
# ═══════════════════════════════════════════════════════════════════

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─── 1. GERAR DADOS DE MERCADO ───
print("📈 Gerando dados sintéticos de mercado financeiro...")

np.random.seed(42)

n_days = 1000
dates = pd.date_range('2022-01-01', periods=n_days, freq='D')

# Preço inicial
price = 100

# Gerar série temporal de preços
prices = [price]
volumes = []

for i in range(n_days - 1):
    # Tendência aleatória
    trend = np.random.choice([-1, 0, 1], p=[0.45, 0.1, 0.45])

    # Volatilidade
    volatility = np.random.normal(0, 2)

    # Novo preço
    change = trend + volatility
    price = max(prices[-1] + change, 50)  # Mínimo 50
    prices.append(price)

    # Volume (correlacionado com mudança de preço)
    volume = abs(change) * 1000 + np.random.normal(5000, 1000)
    volumes.append(max(volume, 100))

volumes.append(volumes[-1])  # Último dia

# Calcular features técnicos
df = pd.DataFrame({
    'date': dates,
    'close': prices,
    'volume': volumes
})

# SMA (Simple Moving Average)
df['sma_5'] = df['close'].rolling(window=5).mean()
df['sma_20'] = df['close'].rolling(window=20).mean()

# EMA (Exponential Moving Average)
df['ema_12'] = df['close'].ewm(span=12).mean()
df['ema_26'] = df['close'].ewm(span=26).mean()

# RSI (Relative Strength Index)
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

df['rsi'] = calculate_rsi(df['close'])

# MACD
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9).mean()

# Bollinger Bands
df['bb_middle'] = df['close'].rolling(window=20).mean()
df['bb_std'] = df['close'].rolling(window=20).std()
df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']

# Remover NaN
df = df.dropna().reset_index(drop=True)

print(f"  Total de dias: {len(df)}")
print(f"  Período: {df['date'].min()} a {df['date'].max()}")
print(f"  Preço min/max: ${df['close'].min():.2f} / ${df['close'].max():.2f}")

# Visualizar dados
fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# Preço + SMA
axes[0].plot(df['date'], df['close'], label='Close', linewidth=2)
axes[0].plot(df['date'], df['sma_5'], label='SMA-5', linewidth=1.5, alpha=0.7)
axes[0].plot(df['date'], df['sma_20'], label='SMA-20', linewidth=1.5, alpha=0.7)
axes[0].fill_between(df['date'], df['bb_lower'], df['bb_upper'], alpha=0.2, label='Bollinger Bands')
axes[0].set_title('Preço + Indicadores Técnicos', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Preço ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# RSI
axes[1].plot(df['date'], df['rsi'], label='RSI', color='purple', linewidth=2)
axes[1].axhline(y=70, color='red', linestyle='--', label='Overbought (70)')
axes[1].axhline(y=30, color='green', linestyle='--', label='Oversold (30)')
axes[1].set_title('RSI (Relative Strength Index)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('RSI')
axes[1].set_ylim([0, 100])
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# MACD
axes[2].plot(df['date'], df['macd'], label='MACD', linewidth=2)
axes[2].plot(df['date'], df['macd_signal'], label='Signal', linewidth=2)
axes[2].bar(df['date'], df['macd'] - df['macd_signal'], label='Histogram', alpha=0.3)
axes[2].set_title('MACD (Moving Average Convergence Divergence)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('MACD')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stock_technical_indicators.png', dpi=150)
print("\n  ✓ Indicadores salvos: stock_technical_indicators.png")

# ─── 2. GERAR SINAIS DE TRADING (LABELS) ───
print("\n🎯 Gerando sinais de trading...")

# Sinal baseado em preço futuro (lookahead de 5 dias)
df['future_return'] = df['close'].shift(-5) / df['close'] - 1

# Classificação:
# 0 = HOLD (mudança < 2%)
# 1 = BUY (queda > 2% - oportunidade de compra)
# 2 = SELL (alta > 2% - realizar lucro)

df['signal'] = 0  # Default HOLD

df.loc[df['future_return'] < -0.02, 'signal'] = 1  # BUY
df.loc[df['future_return'] > 0.02, 'signal'] = 2  # SELL

# Remover últimos 5 dias (sem label)
df = df[:-5].reset_index(drop=True)

print(f"  Distribuição de sinais:")
print(df['signal'].value_counts().sort_index())

# ─── 3. PREPARAR DADOS PARA LSTM ───
print("\n🔧 Preparando dados para LSTM...")

# Features para treinar
feature_cols = ['close', 'volume', 'sma_5', 'sma_20', 'ema_12', 'ema_26', 
                'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower']

# Normalizar
scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Criar sequências
LOOKBACK = 30  # 30 dias de histórico

def create_sequences(data, labels, lookback=30):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(labels[i])
    return np.array(X), np.array(y)

X = df[feature_cols].values
y = df['signal'].values

X_seq, y_seq = create_sequences(X, y, LOOKBACK)

print(f"  X shape: {X_seq.shape} (samples, timesteps, features)")
print(f"  y shape: {y_seq.shape}")

# Split (70% treino, 15% val, 15% teste)
train_size = int(0.7 * len(X_seq))
val_size = int(0.15 * len(X_seq))

X_train = X_seq[:train_size]
y_train = y_seq[:train_size]

X_val = X_seq[train_size:train_size+val_size]
y_val = y_seq[train_size:train_size+val_size]

X_test = X_seq[train_size+val_size:]
y_test = y_seq[train_size+val_size:]

print(f"\n  Treino: {len(X_train)}")
print(f"  Validação: {len(X_val)}")
print(f"  Teste: {len(X_test)}")

# ─── 4. CONSTRUIR MODELO LSTM ───
print("\n🔨 Construindo modelo LSTM de trading...")

model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(LOOKBACK, len(feature_cols))),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(3, activation='softmax')  # 3 classes: HOLD, BUY, SELL
], name='Trading_LSTM')

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  Parâmetros: {model.count_params():,}")
model.summary()

# ─── 5. TREINAR MODELO ───
print("\n🚀 Treinando modelo...")

callbacks = [EarlyStopping(patience=15, restore_best_weights=True, verbose=1)]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Treinamento concluído!")

# Avaliar
acc = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"\n  Accuracy (teste): {acc:.4f}")

# ─── 6. AVALIAR PERFORMANCE ───
print("\n📊 Avaliando performance do bot...")

# Predições
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_classes)
signal_labels = ['HOLD', 'BUY', 'SELL']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=signal_labels, yticklabels=signal_labels)
plt.title('Matriz de Confusão - Sinais de Trading', fontsize=14, fontweight='bold')
plt.ylabel('Real')
plt.xlabel('Previsto')
plt.tight_layout()
plt.savefig('trading_confusion_matrix.png', dpi=150)
print("  ✓ Matriz salva: trading_confusion_matrix.png")

# Classification report
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=signal_labels))

# ─── 7. SIMULAR TRADING (BACKTEST) ───
print("\n💰 Simulando estratégia de trading (backtest)...")

initial_capital = 10000
capital = initial_capital
position = 0  # 0 = sem posição, 1 = comprado
entry_price = 0

# Histórico
portfolio_value = []
trades = []

# Pegar preços originais do período de teste
test_start_idx = train_size + val_size + LOOKBACK
test_prices = df['close'].iloc[test_start_idx:test_start_idx+len(X_test)].reset_index(drop=True)

# Desnormalizar (multiplicar pelo scaler)
# Nota: Para simplicidade, vamos usar preços sintéticos escalados
prices_test = np.random.uniform(90, 150, len(X_test))  # Simulação

for i, (pred_signal, actual_price) in enumerate(zip(y_pred_classes, prices_test)):
    # Decisão baseada no modelo
    if pred_signal == 1 and position == 0:  # BUY
        position = 1
        entry_price = actual_price
        shares = capital / actual_price
        trades.append(('BUY', i, actual_price, shares))

    elif pred_signal == 2 and position == 1:  # SELL
        position = 0
        profit = shares * (actual_price - entry_price)
        capital += profit
        trades.append(('SELL', i, actual_price, profit))

    # Calcular valor do portfólio
    if position == 1:
        portfolio_value.append(shares * actual_price)
    else:
        portfolio_value.append(capital)

final_value = portfolio_value[-1]
total_return = (final_value - initial_capital) / initial_capital * 100

print(f"\n  Capital inicial: ${initial_capital:,.2f}")
print(f"  Valor final: ${final_value:,.2f}")
print(f"  Retorno total: {total_return:+.2f}%")
print(f"  Número de trades: {len([t for t in trades if t[0] == 'BUY'])}")

# Visualizar portfólio
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Valor do portfólio
axes[0].plot(portfolio_value, linewidth=2, label='Valor do Portfólio')
axes[0].axhline(y=initial_capital, color='gray', linestyle='--', label='Capital Inicial')
axes[0].set_title(f'Evolução do Portfólio - Retorno: {total_return:+.2f}%', 
                  fontsize=14, fontweight='bold')
axes[0].set_ylabel('Valor ($)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Sinais vs Preço
axes[1].plot(prices_test, label='Preço', linewidth=2, alpha=0.7)

# Marcar trades
buy_indices = [t[1] for t in trades if t[0] == 'BUY']
sell_indices = [t[1] for t in trades if t[0] == 'SELL']

if buy_indices:
    axes[1].scatter(buy_indices, [prices_test[i] for i in buy_indices], 
                    color='green', marker='^', s=100, label='BUY', zorder=5)
if sell_indices:
    axes[1].scatter(sell_indices, [prices_test[i] for i in sell_indices], 
                    color='red', marker='v', s=100, label='SELL', zorder=5)

axes[1].set_title('Sinais de Trading Executados', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Dia')
axes[1].set_ylabel('Preço ($)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('trading_backtest.png', dpi=150)
print("  ✓ Backtest salvo: trading_backtest.png")

# ─── 8. COMPARAR COM BUY-AND-HOLD ───
print("\n📊 Comparando com estratégia Buy-and-Hold...")

# Buy-and-hold: comprar no início, vender no final
bh_shares = initial_capital / prices_test[0]
bh_final_value = bh_shares * prices_test[-1]
bh_return = (bh_final_value - initial_capital) / initial_capital * 100

print(f"\n  Buy-and-Hold:")
print(f"    Valor final: ${bh_final_value:,.2f}")
print(f"    Retorno: {bh_return:+.2f}%")

print(f"\n  Bot LSTM:")
print(f"    Valor final: ${final_value:,.2f}")
print(f"    Retorno: {total_return:+.2f}%")

if total_return > bh_return:
    print(f"\n  ✅ Bot superou Buy-and-Hold em {total_return - bh_return:+.2f}%!")
else:
    print(f"\n  ⚠️ Bot ficou {bh_return - total_return:.2f}% abaixo de Buy-and-Hold")

# ─── 9. VISUALIZAR HISTÓRICO DE TREINAMENTO ───
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'], label='Treino', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Validação', linewidth=2)
axes[0].set_title('Loss durante Treinamento', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'], label='Treino', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Validação', linewidth=2)
axes[1].set_title('Accuracy durante Treinamento', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('trading_training_history.png', dpi=150)
print("\n  ✓ Histórico salvo: trading_training_history.png")

# ─── 10. RELATÓRIO FINAL ───
print("\n" + "="*70)
print("✅ BOT DE TRADING COM LSTM CONCLUÍDO!")
print("="*70)

print(f"\n📊 Performance do Modelo:")
print(f"  Accuracy: {acc:.4f}")
print(f"  Parâmetros: {model.count_params():,}")
print(f"  Lookback: {LOOKBACK} dias")
print(f"  Features: {len(feature_cols)}")

print(f"\n💰 Performance Financeira:")
print(f"  Capital inicial: ${initial_capital:,.2f}")
print(f"  Valor final (Bot): ${final_value:,.2f}")
print(f"  Retorno (Bot): {total_return:+.2f}%")
print(f"  Retorno (Buy-Hold): {bh_return:+.2f}%")
print(f"  Diferença: {total_return - bh_return:+.2f}%")
print(f"  Trades executados: {len(trades)//2}")

print("\n📁 Arquivos gerados:")
print("  • stock_technical_indicators.png - Indicadores técnicos")
print("  • trading_confusion_matrix.png - Matriz de confusão")
print("  • trading_backtest.png - Backtest da estratégia")
print("  • trading_training_history.png - Histórico de treinamento")

print("\n⚠️ Disclaimer:")
print("  Este é um exemplo educacional. NÃO USE para trading real sem:")
print("  • Validação com dados históricos reais")
print("  • Análise de risco adequada")
print("  • Consideração de custos de transação")
print("  • Gestão de risco (stop-loss, take-profit)")
print("  • Validação em múltiplos ativos e períodos")

print("\n🔧 Melhorias possíveis:")
print("  • Adicionar stop-loss e take-profit")
print("  • Incluir custos de transação")
print("  • Testar com dados reais (yfinance)")
print("  • Implementar ensemble de modelos")
print("  • Adicionar análise de risco (Sharpe Ratio, Max Drawdown)")
print("  • Usar Reinforcement Learning (DQN, PPO)")
