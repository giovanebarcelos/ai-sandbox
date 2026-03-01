# GO1252-Sklearn
from sklearn.metrics import classification_report, confusion_matrix

# Classification Report


if __name__ == "__main__":
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
