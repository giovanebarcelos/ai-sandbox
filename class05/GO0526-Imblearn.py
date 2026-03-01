# GO0526-Imblearn
from imblearn.over_sampling import RandomOverSampler


if __name__ == "__main__":
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)

    # ANTES: [9500 classe 0, 500 classe 1]
    # DEPOIS: [9500 classe 0, 9500 classe 1]
