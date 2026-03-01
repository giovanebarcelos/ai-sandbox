# GO0533-Codigo
# ❌ MAU:


if __name__ == "__main__":
    print(f"Acurácia: {accuracy_score(y_test, y_pred)}")

    # ✅ BOM:
    print(classification_report(y_test, y_pred))
    print(f"F1-Score: {f1_score(y_test, y_pred)}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba)}")
