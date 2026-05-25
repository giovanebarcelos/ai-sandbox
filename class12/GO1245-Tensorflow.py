# GO1245-Tensorflow
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass


if __name__ == '__main__':
    # model.add(Conv2D(64, (3,3), activation='relu'))
    # model.add(BatchNormalization())  # Após Conv, antes/após ReLU

    # ─── VISUALIZAÇÃO: EFEITO DO BATCH NORMALIZATION ───
    np.random.seed(42)
    n = 500

    # Simular ativações sem BN (distribuição arbitrária)
    sem_bn = np.random.randn(n) * 8 + 15    # média=15, std=8 (covariata shift)
    # Após BN: normalizado (μ=0, σ=1) e depois escalonado (gamma=2, beta=1)
    gamma, beta_param = 2.0, 1.0
    bn_normalizado = (sem_bn - sem_bn.mean()) / (sem_bn.std() + 1e-8)
    com_bn = gamma * bn_normalizado + beta_param

    # Simular distribuições em diferentes camadas (internal covariate shift)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Histogramas: sem vs com BN
    axes[0].hist(sem_bn,  bins=40, color='#e15759', alpha=0.7, label=f'Sem BN: μ={sem_bn.mean():.1f}, σ={sem_bn.std():.1f}', density=True)
    axes[0].hist(com_bn,  bins=40, color='#4e79a7', alpha=0.7, label=f'Com BN: μ={com_bn.mean():.2f}, σ={com_bn.std():.2f}', density=True)
    axes[0].axvline(sem_bn.mean(), color='#e15759', linestyle='--', linewidth=2)
    axes[0].axvline(com_bn.mean(), color='#4e79a7', linestyle='--', linewidth=2)
    axes[0].set_title('Distribuição das Ativações\nSem vs Com Batch Normalization', fontsize=11)
    axes[0].set_xlabel('Valor da Ativação')
    axes[0].set_ylabel('Densidade')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Covariata shift: distribuições derivam entre camadas sem BN
    camadas = range(1, 7)
    means_sem = np.array([2, 5, 12, 3, 18, 8])    # deriva aleatoriamente
    stds_sem  = np.array([1, 3, 8, 2, 6, 4])
    means_com = np.zeros(6)
    stds_com  = np.ones(6)

    axes[1].errorbar(camadas, means_sem, yerr=stds_sem, fmt='o-', color='#e15759', linewidth=2,
                     capsize=5, label='Sem BN (média±std)')
    axes[1].errorbar(camadas, means_com, yerr=stds_com, fmt='s-', color='#4e79a7', linewidth=2,
                     capsize=5, label='Com BN (μ≈0, σ≈1)')
    axes[1].axhline(0, color='gray', linestyle=':', linewidth=1)
    axes[1].set_xlabel('Camada')
    axes[1].set_ylabel('Média das Ativações')
    axes[1].set_title('Internal Covariate Shift\nSem BN as distribuições derivam', fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Impacto no treinamento: convergencia mais rápida
    epochs = np.arange(1, 31)
    np.random.seed(7)
    loss_sem_bn = 2.5 * np.exp(-0.08 * epochs) + 0.35 + 0.05 * np.random.randn(30)
    loss_com_bn = 2.5 * np.exp(-0.22 * epochs) + 0.12 + 0.02 * np.random.randn(30)

    axes[2].plot(epochs, loss_sem_bn, 'o-', color='#e15759', linewidth=2, markersize=3,
                 label='Sem BatchNorm')
    axes[2].plot(epochs, loss_com_bn, 's-', color='#4e79a7', linewidth=2, markersize=3,
                 label='Com BatchNorm')
    axes[2].set_xlabel('Época')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Convêrgência do Treinamento\nBN acelera e estabiliza', fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.suptitle('Batch Normalization — Efeito sobre Distribuições e Convergência',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
