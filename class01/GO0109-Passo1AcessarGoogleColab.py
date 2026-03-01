# GO0109-Passo1AcessarGoogleColab
# Verificar ambiente
import sys
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt


if __name__ == "__main__":
    print(f"✅ Python: {sys.version.split()[0]}")
    print(f"✅ NumPy: {np.__version__}")
    print(f"✅ Pandas: {pd.__version__}")
    print(f"✅ Scikit-learn: {sklearn.__version__}")
    print("\n🚀 Ambiente pronto para IA!")
