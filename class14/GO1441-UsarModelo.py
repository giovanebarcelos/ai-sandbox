# GO1441-UsarModelo
# 1. Usar modelo mais estável (Bidirectional NÃO funciona para recursivo!)
# 2. Limitar horizonte de previsão (max 3-5 dias)
# 3. Clip predições no range esperado

if __name__ == "__main__":
    next_pred = np.clip(next_pred, 0, 1)  # Manter em [0,1] escalonado
