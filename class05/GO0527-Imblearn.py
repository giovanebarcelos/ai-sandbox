# GO0527-Imblearn
from imblearn.under_sampling import RandomUnderSampler


if __name__ == "__main__":
    rus = RandomUnderSampler(random_state=42)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    # ANTES: [9500 classe 0, 500 classe 1]
    # DEPOIS: [500 classe 0, 500 classe 1]
