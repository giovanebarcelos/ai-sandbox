# GO0537-Imblearn
from imblearn.over_sampling import SVMSMOTE, ADASYN, BorderlineSMOTE

# SMOTE com SVM


if __name__ == "__main__":
    svm_smote = SVMSMOTE(random_state=42)

    # Adaptive Synthetic (ADASYN)
    adasyn = ADASYN(random_state=42)

    # Borderline SMOTE (foca em amostras na fronteira)
    borderline_smote = BorderlineSMOTE(random_state=42)
