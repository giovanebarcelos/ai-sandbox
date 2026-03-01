# GO1312-LearningRateScheduling
# Learning rate scheduling


if __name__ == "__main__":
    lr0=0.01       # Initial LR
    lrf=0.01       # Final LR (lr0 * lrf)

    # Optimizer
    optimizer='Adam'  # ou 'SGD', 'AdamW'

    # Regularização
    weight_decay=0.0005
