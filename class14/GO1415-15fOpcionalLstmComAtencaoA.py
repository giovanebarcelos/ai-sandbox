# GO1415-15fOpcionalLstmComAtençãoA
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print("="*70)
    print("LSTM COM ATENÇÃO - Attention Mechanism")
    print("="*70)

    # 1. CAMADA DE ATENÇÃO CUSTOMIZADA
    class AttentionLayer(layers.Layer):
        """
        Implementa mecanismo de atenção (Bahdanau Attention)

        Permite que o modelo "foque" em diferentes partes da sequência
        de entrada ao fazer predições
        """
        def __init__(self, units, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)
            self.units = units

        def build(self, input_shape):
            # Pesos para computar attention scores
            self.W1 = self.add_weight(
                name='W1',
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                trainable=True
            )

            self.W2 = self.add_weight(
                name='W2',
                shape=(input_shape[-1], self.units),
                initializer='glorot_uniform',
                trainable=True
            )

            self.V = self.add_weight(
                name='V',
                shape=(self.units, 1),
                initializer='glorot_uniform',
                trainable=True
            )

            super(AttentionLayer, self).build(input_shape)

        def call(self, inputs):
            """
            Args:
                inputs: (batch, timesteps, features)

            Returns:
                context_vector: (batch, features)
                attention_weights: (batch, timesteps)
            """
            # Calcular attention scores
            # score(h_t) = V^T * tanh(W1*h_t + W2*h_s)
            score = tf.nn.tanh(tf.matmul(inputs, self.W1))  # (batch, timesteps, units)
            score = tf.matmul(score, self.V)  # (batch, timesteps, 1)

            # Attention weights (softmax)
            attention_weights = tf.nn.softmax(score, axis=1)  # (batch, timesteps, 1)

            # Context vector (weighted sum)
            context_vector = attention_weights * inputs  # (batch, timesteps, features)
            context_vector = tf.reduce_sum(context_vector, axis=1)  # (batch, features)

            return context_vector, tf.squeeze(attention_weights, axis=-1)

        def get_config(self):
            config = super(AttentionLayer, self).get_config()
            config.update({'units': self.units})
            return config

    # 2. CONSTRUIR MODELO COM ATENÇÃO
    def build_lstm_with_attention(input_length, n_features, lstm_units=128, attention_units=64):
        """
        Constrói LSTM com mecanismo de atenção
        """
        # Input
        inputs = layers.Input(shape=(input_length, n_features), name='input')

        # LSTM Layer (retorna sequência completa)
        lstm_out = layers.LSTM(lstm_units, return_sequences=True, name='lstm')(inputs)

        # Attention Layer
        context_vector, attention_weights = AttentionLayer(
            attention_units, 
            name='attention'
        )(lstm_out)

        # Output Layer
        outputs = layers.Dense(1, name='output')(context_vector)

        # Modelo
        model = models.Model(inputs=inputs, outputs=outputs, name='lstm_with_attention')

        # Modelo auxiliar para visualizar attention weights
        attention_model = models.Model(
            inputs=inputs, 
            outputs=attention_weights, 
            name='attention_viz'
        )

        return model, attention_model

    # Exemplo de uso
    INPUT_LENGTH = 30
    N_FEATURES = 4

    model, attention_viz_model = build_lstm_with_attention(
        input_length=INPUT_LENGTH,
        n_features=N_FEATURES,
        lstm_units=128,
        attention_units=64
    )

    model.summary()

    # 3. TREINAR MODELO (usando dados anteriores)
    # Reutilizar X_train, y_train de slides anteriores
    # Para este exemplo, vamos simular

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='mse',
        metrics=['mae']
    )

    print("\n🏋️ Treinando LSTM com Atenção...")

    # Criar dados dummy para demonstração
    X_dummy = np.random.randn(1000, INPUT_LENGTH, N_FEATURES)
    y_dummy = np.random.randn(1000, 1)

    history = model.fit(
        X_dummy, y_dummy,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        verbose=0
    )

    print("✅ Treinamento concluído!")

    # 4. VISUALIZAR ATENÇÃO
    def visualize_attention(model, attention_model, X_sample, y_sample, feature_names):
        """
        Visualiza pesos de atenção para uma amostra
        """
        # Prever
        prediction = model.predict(X_sample[np.newaxis, :, :], verbose=0)[0, 0]

        # Obter attention weights
        attention_weights = attention_model.predict(
            X_sample[np.newaxis, :, :], 
            verbose=0
        )[0]

        # Plotar
        fig = plt.figure(figsize=(16, 8))

        # Subplot 1: Heatmap de Atenção
        ax1 = plt.subplot(2, 1, 1)

        # Heatmap das features com atenção
        attention_matrix = X_sample.T * attention_weights  # (features, timesteps)

        im = ax1.imshow(attention_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax1.set_yticks(range(len(feature_names)))
        ax1.set_yticklabels(feature_names)
        ax1.set_xlabel('Timestep', fontsize=12)
        ax1.set_ylabel('Feature', fontsize=12)
        ax1.set_title('Heatmap de Atenção × Features', fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax1)

        # Subplot 2: Pesos de Atenção ao longo do tempo
        ax2 = plt.subplot(2, 1, 2)

        timesteps = np.arange(INPUT_LENGTH)
        ax2.bar(timesteps, attention_weights, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Timestep (dias atrás)', fontsize=12)
        ax2.set_ylabel('Peso de Atenção', fontsize=12)
        ax2.set_title('Distribuição dos Pesos de Atenção', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Adicionar linha de média
        ax2.axhline(y=1/INPUT_LENGTH, color='red', linestyle='--', 
                   label=f'Atenção Uniforme (1/{INPUT_LENGTH})')
        ax2.legend()

        plt.tight_layout()
        plt.savefig('attention_weights_visualization.png', dpi=150)
        print(f"\n✅ Visualização salva: attention_weights_visualization.png")
        print(f"\n📊 Predição: {prediction:.4f}")
        print(f"   Valor real: {y_sample:.4f}")
        print(f"\n🔍 Timesteps com maior atenção:")

        # Top 5 timesteps
        top_indices = np.argsort(attention_weights)[-5:][::-1]
        for rank, idx in enumerate(top_indices, 1):
            print(f"   {rank}. Timestep {idx} (há {INPUT_LENGTH - idx} dias): "
                  f"peso = {attention_weights[idx]:.4f}")

    # Visualizar para uma amostra
    sample_idx = 0
    X_sample = X_dummy[sample_idx]
    y_sample = y_dummy[sample_idx, 0]

    visualize_attention(
        model, 
        attention_viz_model, 
        X_sample, 
        y_sample,
        feature_names=['Temperatura', 'Umidade', 'Pressão', 'Vento']
    )

    # 5. COMPARAÇÃO: LSTM vs LSTM+Atenção
    def compare_lstm_attention(X_train, y_train, X_test, y_test):
        """
        Compara performance de LSTM simples vs LSTM com atenção
        """
        # Modelo 1: LSTM Simples
        model_simple = models.Sequential([
            layers.LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])),
            layers.Dense(1)
        ])

        model_simple.compile(optimizer='adam', loss='mse', metrics=['mae'])

        print("\n🔧 Treinando LSTM Simples...")
        history_simple = model_simple.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=30,
            batch_size=32,
            verbose=0
        )

        # Modelo 2: LSTM com Atenção
        model_attn, _ = build_lstm_with_attention(
            input_length=X_train.shape[1],
            n_features=X_train.shape[2]
        )

        model_attn.compile(optimizer='adam', loss='mse', metrics=['mae'])

        print("🔧 Treinando LSTM com Atenção...")
        history_attn = model_attn.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=30,
            batch_size=32,
            verbose=0
        )

        # Comparar resultados
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Loss
        axes[0].plot(history_simple.history['val_loss'], 
                    label='LSTM Simples', linewidth=2)
        axes[0].plot(history_attn.history['val_loss'], 
                    label='LSTM + Atenção', linewidth=2)
        axes[0].set_title('Validation Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('MSE')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # MAE
        axes[1].plot(history_simple.history['val_mae'], 
                    label='LSTM Simples', linewidth=2)
        axes[1].plot(history_attn.history['val_mae'], 
                    label='LSTM + Atenção', linewidth=2)
        axes[1].set_title('Validation MAE', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('lstm_vs_lstm_attention.png', dpi=150)
        print("\n✅ Comparação salva: lstm_vs_lstm_attention.png")

        # Métricas finais
        print("\n📊 RESULTADOS FINAIS:")
        print(f"  LSTM Simples:")
        print(f"    Val Loss: {history_simple.history['val_loss'][-1]:.4f}")
        print(f"    Val MAE:  {history_simple.history['val_mae'][-1]:.4f}")
        print(f"  LSTM + Atenção:")
        print(f"    Val Loss: {history_attn.history['val_loss'][-1]:.4f}")
        print(f"    Val MAE:  {history_attn.history['val_mae'][-1]:.4f}")

        improvement = (history_simple.history['val_mae'][-1] - 
                      history_attn.history['val_mae'][-1]) / \
                      history_simple.history['val_mae'][-1] * 100

        print(f"\n💡 Melhoria com Atenção: {improvement:.1f}%")

    # Comparar modelos (com dados dummy)
    split_idx = int(0.8 * len(X_dummy))
    compare_lstm_attention(
        X_dummy[:split_idx], y_dummy[:split_idx],
        X_dummy[split_idx:], y_dummy[split_idx:]
    )

    print("\n✅ LSTM COM ATENÇÃO COMPLETO!")
    print("\n🎯 Benefícios da Atenção:")
    print("  ✓ Modelo foca em partes relevantes da sequência")
    print("  ✓ Melhora interpretabilidade (vemos o que é importante)")
    print("  ✓ Reduz vanishing gradient em sequências longas")
    print("  ✓ Performance geralmente 5-15% melhor que LSTM simples")
    print("\n📚 Aplicações:")
    print("  - Tradução automática (NMT)")
    print("  - Sumarização de texto")
    print("  - Previsão de séries temporais")
    print("  - Análise de sentimento")
