# GO0523-PandasXgboost
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# Carregar dados Titanic


if __name__ == "__main__":
    train = pd.read_csv('titanic_train.csv')

    # Feature engineering
    train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
    train['IsAlone'] = (train['FamilySize'] == 1).astype(int)
    train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    # Encoding
    le = LabelEncoder()
    train['Sex'] = le.fit_transform(train['Sex'])
    train['Embarked'] = train['Embarked'].fillna('S')
    train['Embarked'] = le.fit_transform(train['Embarked'])

    # Features
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
    X = train[features].fillna(-1)  # XGBoost suporta -1 para missing!
    y = train['Survived']

    # XGBoost com hiperparâmetros otimizados
    model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    )

    # Cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Acurácia média: {scores.mean():.3f} (+/- {scores.std():.3f})")

    # Treinar modelo final
    model.fit(X, y)

    # Feature importance
    importances = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(importances)
