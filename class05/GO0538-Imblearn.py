# GO0538-Imblearn
from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier

# Random Forest com undersampling automático


if __name__ == "__main__":
    brf = BalancedRandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Easy Ensemble (AdaBoost + Undersampling)
    eec = EasyEnsembleClassifier(
        n_estimators=10,
        random_state=42
    )
