# GO0535-Sklearn
from sklearn.model_selection import StratifiedKFold, cross_val_score


if __name__ == "__main__":
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    print(f"F1-Score médio: {scores.mean():.3f} ± {scores.std():.3f}")
