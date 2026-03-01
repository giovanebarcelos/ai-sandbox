# GO2021-PytestPickle
# tests/test_model.py
import pytest
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

def test_model_loading():
    """Testar se modelo pode ser carregado"""
    with open('models/latest.pkl', 'rb') as f:
        model = pickle.load(f)
    assert model is not None, "Modelo não carregou"

def test_model_prediction_shape():
    """Testar shape das predições"""
    with open('models/latest.pkl', 'rb') as f:
        model = pickle.load(f)

    X_test = np.random.rand(10, 4)  # 10 amostras, 4 features
    predictions = model.predict(X_test)

    assert predictions.shape[0] == 10, "Shape incorreto"

def test_model_accuracy_threshold():
    """Testar acurácia mínima"""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.7)
    args, _ = parser.parse_known_args()

    # Carregar modelo e dados de teste
    with open('models/latest.pkl', 'rb') as f:
        model = pickle.load(f)

    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')

    # Predições
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    assert accuracy >= args.threshold, \
        f"Acurácia {accuracy:.3f} abaixo do threshold {args.threshold}"

    print(f"✅ Acurácia: {accuracy:.3f} (threshold: {args.threshold})")

def test_model_fairness():
    """Testar viés em grupos protegidos"""
    with open('models/latest.pkl', 'rb') as f:
        model = pickle.load(f)

    # Carregar dados com atributo sensível (ex: gênero)
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    sensitive_attr = np.load('data/gender_test.npy')  # 0=M, 1=F

    # Predições
    y_pred = model.predict(X_test)

    # Acurácia por grupo
    acc_group0 = accuracy_score(y_test[sensitive_attr==0], 
                                 y_pred[sensitive_attr==0])
    acc_group1 = accuracy_score(y_test[sensitive_attr==1], 
                                 y_pred[sensitive_attr==1])

    # Disparate impact (diferença máxima aceitável: 10%)
    disparity = abs(acc_group0 - acc_group1)
    assert disparity < 0.10, \
        f"Disparidade muito alta: {disparity:.3f} (Grupo0: {acc_group0:.3f}, Grupo1: {acc_group1:.3f})"

    print(f"✅ Fairness OK - Disparidade: {disparity:.3f}")

def test_model_inference_time():
    """Testar latência de inferência"""
    import time

    with open('models/latest.pkl', 'rb') as f:
        model = pickle.load(f)

    X_sample = np.random.rand(1, 4)

    # Medir tempo de 100 predições
    start = time.time()
    for _ in range(100):
        _ = model.predict(X_sample)
    elapsed = time.time() - start

    avg_time_ms = (elapsed / 100) * 1000

    # Threshold: 50ms por predição
    assert avg_time_ms < 50, \
        f"Inferência muito lenta: {avg_time_ms:.2f}ms (max: 50ms)"

    print(f"✅ Latência: {avg_time_ms:.2f}ms")


if __name__ == '__main__':
    import os
    import numpy as np
    import pickle
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    print("=== Suite de Testes de Modelo ML (execução direta) ===")
    print()

    # Preparar artefatos para os testes
    os.makedirs('models', exist_ok=True)
    os.makedirs('data',   exist_ok=True)

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Adicionar atributo sensível fictício para teste de fairness
    gender = np.random.randint(0, 2, len(y_test))

    # Treinar e salvar modelo
    clf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train, y_train)
    with open('models/latest.pkl', 'wb') as f:
        pickle.dump(clf, f)

    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)
    np.save('data/gender_test.npy', gender)

    # Executar cada função de teste
    tests = [
        ("Carregamento do modelo",    test_model_loading),
        ("Shape das predições",        test_model_prediction_shape),
        ("Threshold de acurácia",      test_model_accuracy_threshold),
        ("Fairness (viés)",            test_model_fairness),
        ("Latência de inferência",     test_model_inference_time),
    ]

    passed, failed = 0, 0
    for nome, fn in tests:
        try:
            fn()
            print(f"  ✅ PASS — {nome}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAIL — {nome}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ⚠️  ERROR — {nome}: {e}")
            failed += 1

    print(f"\n  Resultado: {passed} passou / {failed} falhou")

    # Limpar artefatos criados
    for f in ['models/latest.pkl', 'data/X_test.npy',
              'data/y_test.npy', 'data/gender_test.npy']:
        if os.path.exists(f):
            os.remove(f)
    for d in ['models', 'data']:
        try:
            os.rmdir(d)
        except OSError:
            pass
