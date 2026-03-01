# GO2006-10FastapiSalvarERodarModelo
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import pickle

X, y = load_iris(return_X_y=True)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)
