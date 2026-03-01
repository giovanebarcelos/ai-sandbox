# GO0112-Experimentar
# Testar com nova flor
nova_flor = [[5.0, 3.6, 1.4, 0.2]]
predicao = modelo.predict(nova_flor)
prob = modelo.predict_proba(nova_flor)

print(f"Predição: {iris.target_names[predicao[0]]}")
print("\nProbabilidades:")
for i, p in enumerate(prob[0]):
    print(f"  {iris.target_names[i]}: {p*100:.2f}%")

# Experimentar profundidades
print("\n🔬 Experimento: Diferentes max_depth")
for depth in [1, 2, 3, 5, 10, None]:
    m = DecisionTreeClassifier(max_depth=depth, random_state=42)
    m.fit(X_train, y_train)
    y_p = m.predict(X_test)
    acc = accuracy_score(y_test, y_p)
    print(f"max_depth={depth}: {acc*100:.2f}%")
