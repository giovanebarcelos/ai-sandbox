# GO1644-Codigo


if __name__ == "__main__":
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,        # alpha < 2×r reduz overfitting
        lora_dropout=0.1,     # Dropout ajuda
        # ...
    )

    # Treino
    training_args = TrainingArguments(
        num_train_epochs=3,   # Não treinar muito (2-5 épocas)
        weight_decay=0.01,    # Regularização L2
        # ...
    )
