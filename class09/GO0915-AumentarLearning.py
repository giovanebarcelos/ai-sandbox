  # GO0915-AumentarLearning
  # 1. Aumentar learning rate
  self.lr = 0.1  # Testar: 0.001, 0.01, 0.1, 0.5
  
  # 2. Usar He initialization (já implementado no código)
  w = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
  
  # 3. Verificar normalização
  print(f"X_train min/max: {X_train.min()}/{X_train.max()}")  # Deve estar [-2, 3] após StandardScaler
  ```

**Problema 2: Loss explode (NaN/Inf após algumas iterações)**

- **Sintoma:** Loss vira `NaN` ou `inf` após 5-10 épocas

- **Causa:** Learning rate muito alta OU gradientes explodindo

- **Solução:**
  
  ```python
  # GO0916-ReduzirLearning
  # 1. Reduzir learning rate
  self.lr = 0.01  # Começar conservador
  ```

# 2. Gradient clipping

delta = np.clip(delta, -5, 5)  **# Adicionar no método backward()**

# 3. Verificar Softmax (overflow)

# (Já implementado com trick de estabilidade numérica no código)

**Problema 3: Overfitting severo (Train 99%, Val 70%)**

- **Sintoma:** Train accuracy muito maior que validation accuracy (gap > 20%)

- **Causa:** Modelo muito complexo OU treino muito longo OU sem regularização

- **Solução:**


#### (Já implementado com trick de estabilidade numérica no código)

🔗 [Acessar código no GitHub](https://raw.githubusercontent.com/giovanebarcelos/ai-sandbox/refs/heads/main/class09/GO0917-ReduzirComplexidade.py)


**Problema 4: Código lento (>2 min por época)**

- **Sintoma:** Treinamento demorando muito

- **Causa:** Usando dataset completo (70k) ou batch size muito pequeno

- **Solução:**
  
  ```python
  # GO0918-UsarSubset
  # 1. Usar subset menor
  X_subset = X[:10000]  # Já implementado no código
  ```

# 2. Aumentar batch size

batch_size = 128  **# Ao invés de 32**

**Problema 5: `ValueError: shapes not aligned` (erro de dimensão)**

- **Sintoma:** Erro de matriz durante forward/backward

- **Causa:** Esqueceu de fazer reshape dos dados ou one-hot encoding

- **Solução:**


#### 2. Aumentar batch size

🔗 [Acessar código no GitHub](https://raw.githubusercontent.com/giovanebarcelos/ai-sandbox/refs/heads/main/class09/GO0919-VerificarShapes.py)


# 2. Garantir one-hot encoding

y_train_oh = one_hot_encode(y_train)  # Já implementado no código

**🚀 DESAFIOS OPCIONAIS (PARA ALUNOS AVANÇADOS):**

1. **Early Stopping:** Implementar parada antecipada quando val_loss para de melhorar
   - **Dificuldade:** ⭐⭐
   - **Tempo adicional:** +10 min
   - **Conceito:** Evitar overfitting monitorando validação
   - **Dica:** Salvar best_weights quando val_loss diminui, parar após 10 épocas sem melhora
2. **Batch Normalization:** Normalizar ativações entre camadas
   - **Dificuldade:** ⭐⭐⭐⭐
   - **Tempo adicional:** +20 min
   - **Conceito:** Estabilizar treinamento, permitir learning rates maiores
   - **Paper:** Ioffe & Szegedy (2015) - "Batch Normalization: Accelerating Deep Network Training"
3. **Comparação com Keras:** Implementar mesma arquitetura (784-128-64-10) com TensorFlow/Keras
   - **Dificuldade:** ⭐⭐
   - **Tempo adicional:** +15 min
   - **Objetivo:** Verificar se implementação NumPy pura está correta
   - **Dica:** Accuracy deve ser similar (±2%)
4. **Visualização de Pesos:** Plotar pesos da primeira camada como imagens 28×28
   - **Dificuldade:** ⭐⭐
   - **Tempo adicional:** +10 min
   - **Conceito:** Entender o que neurônios aprendem (detectores de bordas, curvas)
5. **Experimento Ablation:** Testar impacto de diferentes componentes
   - **Dificuldade:** ⭐⭐⭐
   - **Tempo adicional:** +20 min
   - **Comparações:** Xavier vs He initialization, ReLU vs Sigmoid, diferentes learning rates
   - **Documentar:** Criar tabela com resultados

**📚 RECURSOS COMPLEMENTARES:**

**Documentação:**

- [NumPy Broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - Essencial para entender operações matriciais
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) - Detalhes do dataset original
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) - Confusion matrix, classification report

**Papers/Artigos:**

- **LeCun et al. (1998)** - "Gradient-Based Learning Applied to Document Recognition" - Paper original MNIST
- **He et al. (2015)** - "Delving Deep into Rectifiers" - He initialization explicada
- **Glorot & Bengio (2010)** - "Understanding the difficulty of training deep feedforward neural networks" - Xavier initialization

**Vídeos:**

- [3Blue1Brown - Neural Networks Series](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) (60 min) - Visualização geométrica de redes neurais ⭐⭐⭐⭐⭐
- [StatQuest - Softmax & Cross-Entropy](https://www.youtube.com/watch?v=6ArSys5qHAU) (15 min) - Explicação intuitiva

**Livros:**

- **Nielsen, M.** - "Neural Networks and Deep Learning" (online grátis) - Capítulo 3 (Backpropagation)
- **Goodfellow et al.** - "Deep Learning" - Capítulo 6 (Deep Feedforward Networks)

---

**⚙️ CHECKPOINTS DE VALIDAÇÃO POR ETAPA:**

### ✅ Slide 21 (Etapa 1 - Dados):

- [ ] Dataset carregado: `X.shape == (70000, 784)`
- [ ] Normalização OK: `X.min() == 0, X.max() == 1`
- [ ] Split correto: Train: 6000, Val: 2000, Test: 2000
- [ ] One-hot encoding: `y_train_oh.shape == (6000, 10)`
- [ ] Visualização mostra dígitos claros (sem ruído excessivo)

### ✅ Slide 22 (Etapa 2 - Arquitetura):

- [ ] Classe `MulticlassNN` definida sem erros
- [ ] Arquitetura correta: 784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax)
- [ ] Inicialização He implementada: `np.sqrt(2.0 / n_in)`
- [ ] Métodos `forward()` e `backward()` definidos
- [ ] Softmax estável (sem overflow): `exp_z = np.exp(z - np.max(z))`

### ✅ Slide 23 (Etapa 3 - Treinamento):

- [ ] Modelo treina sem erros (50 épocas)
- [ ] Loss diminuindo consistentemente (não estagnado)
- [ ] Train accuracy > 90% ao final
- [ ] Val accuracy > 88% ao final
- [ ] Gap treino/val < 15% (sem overfitting severo)
- [ ] Curvas de loss convergem (não oscilando demais)

### ✅ Slide 24 (Etapa 4 - Análise):

- [ ] Test accuracy > 85%
- [ ] Confusion matrix gerada e visualizada
- [ ] Classification report mostra F1-score por dígito
- [ ] Top 20 erros mais confiantes plotados
- [ ] Análise por dígito identifica dígitos difíceis (ex: 4, 9)

---

**💡 DICAS GERAIS:**

1. **Execute célula por célula:** Não rode tudo de uma vez, valide cada etapa antes de prosseguir
2. **Monitore shapes:** Use `print(variable.shape)` frequentemente para evitar erros de dimensão
3. **Comece com valores padrão:** Learning rate=0.1, batch_size=64, epochs=50
4. **Experimente depois:** Após funcionar, teste diferentes hiperparâmetros
5. **Salve resultados:** Grave gráficos e métricas para comparações futuras
6. **Documente insights:** Anote observações interessantes (ex: "8 confundido com 3")

---

## Slide 21A: 💻 Prática - MNIST (Parte 1: Setup e Dados)

**⏱️ TEMPO ESTIMADO: 10 minutos**

**📦 Instalação necessária:**

