# GO0908-AnaliseDeErros
# ═══════════════════════════════════════════════════════════════════
# ATIVIDADE PRÁTICA: RECONHECIMENTO DE DÍGITOS (MNIST)
# Etapa 4: ANÁLISE DE ERROS E INSIGHTS
# ═══════════════════════════════════════════════════════════════════
"""
Etapa 4 da atividade MNIST: depois de treinar a rede (GO0914/GO0916),
analisa os erros cometidos pela rede neural treinada.

Pré-requisito: rodar GO0916-MNISTSolucaoCompleta.py antes,
ou executar este arquivo que inclui um setup mínimo autossuficiente.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report


# ───────────────────────────────────────────────────────────────────
# SETUP MÍNIMO (rede MLP simples para gerar predições)
# ───────────────────────────────────────────────────────────────────

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

class MLPMinimo:
    """MLP mínimo para demonstração de análise de erros."""

    def __init__(self, hidden_sizes=(128, 64), lr=0.1, epochs=30, batch=256, seed=42):
        self.hidden_sizes = hidden_sizes
        self.lr = lr
        self.epochs = epochs
        self.batch = batch
        np.random.seed(seed)
        self.Ws = []
        self.bs = []
        self.history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

    def _init_params(self, n_in, n_out):
        sizes = [n_in] + list(self.hidden_sizes) + [n_out]
        for i in range(len(sizes) - 1):
            W = np.random.randn(sizes[i], sizes[i+1]) * np.sqrt(2.0 / sizes[i])
            b = np.zeros((1, sizes[i+1]))
            self.Ws.append(W)
            self.bs.append(b)

    def forward(self, X):
        self._acts = [X]
        a = X
        for i, (W, b) in enumerate(zip(self.Ws, self.bs)):
            z = a @ W + b
            a = relu(z) if i < len(self.Ws) - 1 else softmax(z)
            self._acts.append(a)
        return a

    def _cross_entropy(self, probs, y_oh):
        return -np.mean(np.sum(y_oh * np.log(probs + 1e-9), axis=1))

    def fit(self, X_tr, y_tr, X_val, y_val):
        n_cls = len(np.unique(y_tr))
        self._init_params(X_tr.shape[1], n_cls)
        eye = np.eye(n_cls)

        for ep in range(self.epochs):
            idx = np.random.permutation(len(X_tr))
            for start in range(0, len(X_tr), self.batch):
                bi = idx[start:start+self.batch]
                Xb, yb = X_tr[bi], eye[y_tr[bi]]
                probs = self.forward(Xb)
                # Backward (simplificado)
                dA = probs - yb
                for i in range(len(self.Ws)-1, -1, -1):
                    dW = self._acts[i].T @ dA / len(Xb)
                    db = dA.mean(axis=0, keepdims=True)
                    self.Ws[i] -= self.lr * dW
                    self.bs[i] -= self.lr * db
                    if i > 0:
                        dA = (dA @ self.Ws[i].T) * (self._acts[i] > 0)

            p_tr = self.forward(X_tr);  l_tr = self._cross_entropy(p_tr, eye[y_tr])
            p_vl = self.forward(X_val); l_vl = self._cross_entropy(p_vl, eye[y_val])
            acc_tr = np.mean(p_tr.argmax(1) == y_tr)
            acc_vl = np.mean(p_vl.argmax(1) == y_val)
            self.history['loss'].append(l_tr);     self.history['val_loss'].append(l_vl)
            self.history['acc'].append(acc_tr);    self.history['val_acc'].append(acc_vl)
            if (ep+1) % 10 == 0:
                print(f"  Época {ep+1:3d} | loss {l_tr:.4f} | val_loss {l_vl:.4f}"
                      f" | acc {acc_tr*100:.1f}% | val_acc {acc_vl*100:.1f}%")


# ───────────────────────────────────────────────────────────────────
# ETAPA 4 – ANÁLISE DE ERROS E INSIGHTS
# ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # 1. Dados ─────────────────────────────────────────────────────
    print("Carregando MNIST (subconjunto)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist.data, mnist.target.astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=4000, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # 2. Treinar ───────────────────────────────────────────────────
    print("\nTreinando rede neural (MLP 128→64)...")
    model = MLPMinimo(hidden_sizes=(128, 64), lr=0.1, epochs=30)
    model.fit(X_train, y_train, X_val, y_val)

    probs  = model.forward(X_test)
    y_pred = probs.argmax(axis=1)
    test_acc = np.mean(y_pred == y_test)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")

    # 3. Curvas de treino ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs_range = range(1, len(model.history['loss'])+1)

    axes[0].plot(epochs_range, model.history['loss'],     label='Treino')
    axes[0].plot(epochs_range, model.history['val_loss'], label='Validação')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Curvas de Perda')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range,
                 [a*100 for a in model.history['acc']],     label='Treino')
    axes[1].plot(epochs_range,
                 [a*100 for a in model.history['val_acc']], label='Validação')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Acurácia (%)')
    axes[1].set_title('Curvas de Acurácia')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([50, 100])

    plt.tight_layout()
    plt.savefig('GO0908_curvas_treinamento.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("Salvo: GO0908_curvas_treinamento.png")

    # 4. Matriz de confusão ────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title(f'Matriz de Confusão - MNIST (Acc: {test_acc*100:.2f}%)')
    plt.tight_layout()
    plt.savefig('GO0908_matriz_confusao.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("Salvo: GO0908_matriz_confusao.png")

    # 5. Classification report ────────────────────────────────────
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_test, y_pred,
                                target_names=[str(i) for i in range(10)]))

    # 6. Análise de erros ─────────────────────────────────────────
    errors        = y_test != y_pred
    error_indices = np.where(errors)[0]
    pred_conf     = np.max(probs, axis=1)
    print(f"Total de erros: {errors.sum()} / {len(y_test)}"
          f" ({errors.sum()/len(y_test)*100:.2f}%)")

    # Top erros mais confiantes
    error_conf   = pred_conf[error_indices]
    top20_idx    = error_indices[np.argsort(error_conf)[-20:]]

    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    axes = axes.ravel()
    for i, idx in enumerate(top20_idx):
        img = scaler.inverse_transform(X_test[idx].reshape(1, -1)).reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(
            f"True:{y_test[idx]}, Pred:{y_pred[idx]}\nConf:{pred_conf[idx]:.2f}",
            fontsize=8)
        axes[i].axis('off')
    plt.suptitle('Top 20 Erros Mais Confiantes', fontsize=14)
    plt.tight_layout()
    plt.savefig('GO0908_erros_confiantes.png', dpi=100, bbox_inches='tight')
    plt.show()
    print("Salvo: GO0908_erros_confiantes.png")

    # 7. Acurácia por dígito ──────────────────────────────────────
    print("\n" + "="*60)
    print("ACURÁCIA POR DÍGITO")
    print("="*60)
    digit_accs = []
    for d in range(10):
        mask = y_test == d
        acc_d = np.mean(y_pred[mask] == y_test[mask]) * 100
        digit_accs.append((d, acc_d, mask.sum()))
        print(f"  Dígito {d}: {acc_d:5.1f}%  ({mask.sum()} amostras)")

    # 8. Top 10 confusões mais comuns ─────────────────────────────
    print("\nTop 10 confusões mais comuns:")
    pairs = [(cm[i,j], i, j) for i in range(10) for j in range(10) if i != j]
    pairs.sort(reverse=True)
    for cnt, ti, pi in pairs[:10]:
        print(f"  {ti} → {pi} : {cnt} vezes")

    # 9. Resumo final ─────────────────────────────────────────────
    print("\n" + "="*60)
    easiest = max(digit_accs, key=lambda x: x[1])
    hardest = min(digit_accs, key=lambda x: x[1])
    print(f"Dígito mais fácil : {easiest[0]} ({easiest[1]:.1f}%)")
    print(f"Dígito mais difícil: {hardest[0]} ({hardest[1]:.1f}%)")

    if test_acc >= 0.93:
        nivel = "AVANÇADO (≥93%) 🌟"
    elif test_acc >= 0.90:
        nivel = "INTERMEDIÁRIO (≥90%) ✅"
    elif test_acc >= 0.85:
        nivel = "MÍNIMO (≥85%) ✅"
    else:
        nivel = "ABAIXO DO MÍNIMO (<85%) ⚠️"
    print(f"Nível alcançado: {nivel}")
    print("\n✅ Análise de erros completa!")
