# GO0536-Sklearn
from sklearn.metrics import classification_report


if __name__ == "__main__":
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"F1 Classe 0: {report['0']['f1-score']:.3f}")
    print(f"F1 Classe 1: {report['1']['f1-score']:.3f}")  # ← FOCO AQUI!
