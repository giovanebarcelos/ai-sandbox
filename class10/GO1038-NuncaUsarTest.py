# GO1038-NuncaUsarTest
# NUNCA usar test set durante desenvolvimento!
# Apenas no final, uma vez

# Desenvolvimento: train + val
model.fit(X_train, y_train, validation_data=(X_val, y_val))

# Depois de tudo pronto:
final_score = model.evaluate(X_test, y_test)
