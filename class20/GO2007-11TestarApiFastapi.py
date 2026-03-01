# GO2007-11TestarApiFastapi
import requests


if __name__ == "__main__":
    url = "http://localhost:8000/predict"
    data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = requests.post(url, json=data)
    print(response.json())
    # {'prediction': 'setosa', 'prediction_index': 0, 'probabilities': {...}}
