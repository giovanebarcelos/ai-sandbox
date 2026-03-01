# GO0539-Xgboost
# XGBoost com scale_pos_weight
import xgboost as xgb


if __name__ == "__main__":
    ratio = len(y[y==0]) / len(y[y==1])
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=ratio,  # Ajusta peso automaticamente
        random_state=42
    )
