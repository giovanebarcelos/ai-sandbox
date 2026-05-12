# GO0109-Passo1AcessarGoogleColab
# Verificar ambiente
import sys
import numpy as np
import pandas as pd
import sklearn

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass  # Fora do Colab/Jupyter: plt.show() gerencia o display normalmente

if __name__ == "__main__":
    print(f"✅ Python: {sys.version.split()[0]}")
    print(f"✅ NumPy: {np.__version__}")
    print(f"✅ Pandas: {pd.__version__}")
    print(f"✅ Scikit-learn: {sklearn.__version__}")
    print("\n🚀 Ambiente pronto para IA!")
