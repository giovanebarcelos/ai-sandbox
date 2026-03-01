# GO0522-RandomForest
# Random Forest


if __name__ == "__main__":
    importances = rf.feature_importances_

    # XGBoost (3 métodos)
    xgb.plot_importance(model, importance_type='weight')  # Frequência
    xgb.plot_importance(model, importance_type='gain')    # Qualidade
    xgb.plot_importance(model, importance_type='cover')   # Cobertura
