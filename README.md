# ai-sandbox

Repositório de código-fonte da disciplina **Inteligência Artificial** (2026).

Contém **711 scripts Python** organizados por aula, cobrindo desde os fundamentos de IA até tópicos avançados como LLMs, IA generativa e MLOps.

---

## Estrutura do Repositório

```
repository/
├── class01/   # Aula 01 — Introdução à IA & Ética
├── class02/   # Aula 02 — Representação do Conhecimento
├── class03/   # Aula 03 — Resolução de Problemas & Busca
├── class04/   # Aula 04 — Introdução ao Machine Learning
├── class05/   # Aula 05 — Algoritmos de Classificação
├── class06/   # Aula 06 — Regressão & Validação
├── class07/   # Aula 07 — Clustering
├── class08/   # Aula 08 — SOM — Mapas Auto-Organizáveis
├── class09/   # Aula 09 — Fundamentos de Redes Neurais
├── class10/   # Aula 10 — MLP — Redes Multicamadas
├── class11/   # Aula 11 — Lógica Fuzzy
├── class12/   # Aula 12 — CNNs — Redes Convolucionais
├── class13a/  # Aula 13 — Visão Computacional & Reconhecimento de Padrões
├── class14/   # Aula 14 — RNN / LSTM
├── class15/   # Aula 15 — Introdução ao NLP
├── class16/   # Aula 16 — Transformers & LLMs
├── class17/   # Aula 17 — LLMs Locais, Ollama & RAG
├── class18/   # Aula 18 — Reinforcement Learning
├── class19/   # Aula 19 — Algoritmos Genéticos & Metaheurísticas
├── class20/   # Aula 20 — MLOps & Deploy
└── class21/   # Aula 21 — IA Generativa & Futuro
```

---

## Tecnologias Utilizadas

| Categoria           | Bibliotecas                                     |
| ------------------- | ----------------------------------------------- |
| Machine Learning    | scikit-learn, XGBoost, imbalanced-learn         |
| Deep Learning       | TensorFlow / Keras, PyTorch                     |
| Visão Computacional | OpenCV, Ultralytics (YOLO), Pillow              |
| NLP                 | NLTK, spaCy, Gensim, HuggingFace Transformers   |
| LLMs / RAG          | Ollama, LangChain, FAISS, sentence-transformers |
| Otimização          | DEAP, scipy, CMA-ES                             |
| MLOps               | MLflow, FastAPI, Optuna, Prometheus             |
| Visualização        | Matplotlib, Seaborn, Tensorboard                |
| Fuzzy               | scikit-fuzzy, simpful                           |

---

## Conteúdo por Aula

### Aula 01 — Introdução à IA & Ética `class01/` · 22 scripts

Panorama da IA moderna, comparação de abordagens (simbólica vs conexionista) e dimensões éticas.

| Código                                                                                                                                  | Descrição                                              |
| --------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| [GO0101](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class01/GO0101-SimuladorDoTesteDeTuring.py)                            | Simulador do Teste de Turing                           |
| [GO0102](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class01/GO0102-AnaliseDeViesEmSistemasDeIA.py)                         | Análise de viés em sistemas de IA                      |
| [GO0103](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class01/GO0103-BenchmarksDeInteligenciaArtificial.py)                  | Benchmarks de Inteligência Artificial                  |
| [GO0104](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class01/GO0104-CodigoClassificadorSimples.py)                          | Classificador simples (regras vs ML)                   |
| [GO0105–06](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class01/)                                                           | Programação tradicional por regras vs Machine Learning |
| [GO0107](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class01/GO0107-UsandoModeloPretreinadoMobilenetv2.py)                  | Modelo pré-treinado MobileNetV2                        |
| [GO0108](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class01/GO0108-SistemaDeRecomendacao30Min.py)                          | Sistema de recomendação em 30 minutos                  |
| [GO0113](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class01/GO0113-ExperimentoCapacidadesELimitacoesDeLlms.py)             | Experimento: capacidades e limitações de LLMs          |
| [GO0114](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class01/GO0114-AgenteDeIaAutonomoRoboAspirador.py)                     | Agente autônomo — robô aspirador                       |
| [GO0115](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class01/GO0115-SimuladorDeDilemasEticosEmIa.py)                        | Simulador de dilemas éticos em IA                      |
| [GO0116](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class01/GO0116-ComparacaoIaSimbolicaVsConexionista.py)                 | Comparação IA simbólica vs conexionista                |
| [GO0117](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class01/GO0117-AgiArtificialGeneralIntelligenceAvaliadorInterativo.py) | Avaliador interativo de AGI                            |
| [GO0118](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class01/GO0118-XaiExplainableAiTecnicasDeInterpretabilidade.py)        | XAI — técnicas de interpretabilidade                   |
| [GO0119](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class01/GO0119-ImpactoDaIaNoTrabalhoSimulador.py)                      | Impacto da IA no trabalho (simulador)                  |
| [GO0122](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class01/GO0122-GeradorDeRoadmapPessoalCarreira.py)                     | Gerador de roadmap pessoal de carreira                 |

---

### Aula 02 — Representação do Conhecimento `class02/` · 22 scripts

Sistemas especialistas, motores de inferência, grafos de conhecimento e raciocínio incerto.

| Código                                                                                                                            | Descrição                                      |
| --------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| [GO0201](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class02/GO0201-InovacaoFatoresDeCerteza00A.py)                   | Fatores de certeza (CF)                        |
| [GO0202A](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class02/GO0202A-Neo4jEKnowledgeGraphs.py)                       | Neo4j e Knowledge Graphs                       |
| [GO0203](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class02/GO0203-ObjetivoSistemaEspecialistaQueRecomendaFilmes.py) | Sistema especialista de recomendação de filmes |
| [GO0204–05](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class02/)                                                     | Motor de inferência e execução                 |
| [GO0208](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class02/GO0208-GrafoConhecimentoMedico.py)                       | Grafo de conhecimento médico                   |
| [GO0210](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class02/GO0210-MotorDeInferenciaComForwardChaining.py)           | Motor de inferência com forward chaining       |
| [GO0211](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class02/GO0211-PrologSimplesEmPython.py)                         | Prolog simplificado em Python                  |
| [GO0212](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class02/GO0212-LogicaFuzzyParaDecisoes.py)                       | Lógica fuzzy para decisões                     |
| [GO0213](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class02/GO0213-RaciocinioEstruturado.py)                         | Raciocínio estruturado                         |
| [GO0214](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class02/GO0214-ConstraintSatisfactionProblemsCSP.py)             | CSP — Constraint Satisfaction Problems         |
| [GO0215](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class02/GO0215-STRIPSPlanning.py)                                | STRIPS Planning                                |
| [GO0216](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class02/GO0216-RedesSemanticas.py)                               | Redes semânticas                               |
| [GO0217](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class02/GO0217-Bayes.py)                                         | Raciocínio bayesiano                           |
| [GO0218](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class02/GO0218-ProductionSystems.py)                             | Sistemas de produção                           |

---

### Aula 03 — Resolução de Problemas & Busca `class03/` · 15 scripts

Algoritmos de busca cega e heurística para resolução de problemas em grafos e labirintos.

| Código                                                                                                                | Descrição                                  |
| --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| [GO0301](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class03/GO0301-RepresentacaoDeCadaPassoDaBusca.py)   | Representação de estados de busca          |
| [GO0306](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class03/GO0306-HeuristicasComuns.py)                 | Heurísticas comuns (Manhattan, Euclidiana) |
| [GO0307](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class03/GO0307-24VisualizacaoDoLabirintoESolucao.py) | Visualização de labirinto e solução        |
| [GO0311](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class03/GO0311-BfsVsAAnaliseEmpirica.py)             | Comparação empírica BFS vs A*              |
| [GO0312](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class03/GO0312-BuscaDijkstraGrafoPonderado.py)       | Busca de Dijkstra em grafo ponderado       |
| [GO0313](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class03/GO0313-PuzzleOitoComAEstrela.py)             | Puzzle das 8 peças com A*                  |
| [GO0314](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class03/GO0314-BuscaBidirecional.py)                 | Busca bidirecional                         |
| [GO0327A](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class03/GO0327A-MCTSAlphaGoSimplificado.py)         | MCTS AlphaGo simplificado                  |

---

### Aula 04 — Introdução ao Machine Learning `class04/` · 32 scripts

Fundamentos do ML supervisionado com scikit-learn: pipeline, validação e diagnóstico.

| Código                                                                                                                                 | Descrição                                                            |
| -------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| [GO0403](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class04/GO0403-CompletoClassificacaoDeFloresIris.py)                  | Pipeline completo — flores Iris                                      |
| [GO0404](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class04/GO0404-LearningCurvesDiagnosticoDeOverfittingunderfitting.py) | Learning curves — diagnóstico de overfitting/underfitting            |
| [GO0405](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class04/GO0405-ValidacaoCruzadaComSklearn.py)                         | Validação cruzada com scikit-learn                                   |
| [GO0406](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class04/GO0406-OtimizacaoDeHiperparametrosComGridSearch.py)           | Otimização de hiperparâmetros com GridSearch                         |
| [GO0407](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class04/GO0407-PipelineCompletoDeMachineLearning.py)                  | Pipeline completo de Machine Learning                                |
| [GO0412–20](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class04/)                                                          | Exercícios: EDA, preprocessamento, feature engineering, data leakage |
| [GO0418–20](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class04/)                                                          | Exercícios avançados: pipeline automatizado, curvas de aprendizado   |
| [GO0422](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class04/GO0422-DetectarSpamTradicional.py)                            | Detecção de spam (abordagem tradicional)                             |

---

### Aula 05 — Algoritmos de Classificação `class05/` · 43 scripts

KNN, árvores de decisão, Naive Bayes, métricas de avaliação, ensemble e desbalanceamento.

| Código                                                                                                                   | Descrição                                                       |
| ------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| [GO0501](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class05/GO0501-KnearestNeighborsDoZero.py)              | KNN do zero                                                     |
| [GO0502](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class05/GO0502-ArvoreDeDecisaoComSklearn.py)            | Árvore de decisão com scikit-learn                              |
| [GO0503](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class05/GO0503-NaiveBayesComSklearn.py)                 | Naive Bayes com scikit-learn                                    |
| [GO0504](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class05/GO0504-MatrizDeConfusaoAnalise.py)              | Matriz de confusão e análise                                    |
| [GO0505](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class05/GO0505-RocCurveEAuc.py)                         | ROC Curve e AUC                                                 |
| [GO0506](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class05/GO0506-AjustandoThresholdDeDecisao.py)          | Ajuste de threshold de decisão                                  |
| [GO0507](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class05/GO0507-MetricasParaProblemasMulticlasse.py)     | Métricas para problemas multiclasse                             |
| [GO0508–12](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class05/)                                            | Prática MNIST: preparação, treino, análise de erros, otimização |
| [GO0509](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class05/GO0509-DiagnosticoDiabetesEnsemble.py)          | Diagnóstico de diabetes com Ensemble                            |
| [GO0510](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class05/GO0510-ClassificacaoImagensTransferLearning.py) | Classificação de imagens com Transfer Learning                  |
| [GO0521–27](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class05/)                                            | Random Forest, XGBoost, SMOTE, Imbalanced-learn                 |
| [GO0540](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class05/GO0540-DatasetCredit.py)                        | Dataset de crédito — classificação real                         |

---

### Aula 06 — Regressão & Validação `class06/` · 36 scripts

Regressão linear, polinomial e regularizada; feature engineering e análise de resíduos.

| Código                                                                                                            | Descrição                                                           |
| ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [GO0601](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class06/GO0601-RegressaoLinearEMultipla.py)      | Regressão linear e múltipla                                         |
| [GO0602](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class06/GO0602-RegressaoPolinomial.py)           | Regressão polinomial                                                |
| [GO0603](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class06/GO0603-ValidacaoCruzadaParaRegressao.py) | Validação cruzada para regressão                                    |
| [GO0604](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class06/GO0604-RegularizacaoRidgeL2.py)          | Regularização Ridge (L2)                                            |
| [GO0605](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class06/GO0605-RegularizacaoLassoL1.py)          | Regularização Lasso (L1)                                            |
| [GO0606](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class06/GO0606-ElasticNetL1L2.py)                | ElasticNet (L1 + L2)                                                |
| [GO0607–08](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class06/)                                     | Feature engineering e análise de resíduos                           |
| [GO0609–14](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class06/)                                     | Atividade prática: previsão de preços, EDA, treinamento, validação  |
| [GO0618–21](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class06/)                                     | Coeficientes padronizados, importância de features, learning curves |

---

### Aula 07 — Clustering `class07/` · 14 scripts

Aprendizado não supervisionado: K-Means, DBSCAN e clustering hierárquico.

| Código                                                                                                                 | Descrição                                                             |
| ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| [GO0701](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class07/GO0701-VisualizacaoDoProcesso.py)             | Visualização do processo de clustering                                |
| [GO0702](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class07/GO0702-GraficoDoCotovelo.py)                  | Gráfico do cotovelo (elbow method)                                    |
| [GO0703](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class07/GO0703-DbscanComSklearn.py)                   | DBSCAN com scikit-learn                                               |
| [GO0704](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class07/GO0704-DeCorte.py)                            | Dendrograma de corte                                                  |
| [GO0706](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class07/GO0706-SegmentacaoDeClientesParaMarketing.py) | Segmentação de clientes para marketing                                |
| [GO0707–09](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class07/)                                          | Pipeline: normalização, análise de clusters, estratégias de marketing |

---

### Aula 08 — SOM — Mapas Auto-Organizáveis `class08/` · 15 scripts

Self-Organizing Maps com MiniSom para clustering, visualização e detecção de anomalias.

| Código                                                                                                        | Descrição                                                                 |
| ------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| [GO0801](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class08/GO0801-FasesDoTreinamento.py)        | Fases do treinamento SOM                                                  |
| [GO0802](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class08/GO0802-8DeUsoCoresRgb.py)            | Uso de cores/canais RGB                                                   |
| [GO0805](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class08/GO0805-11SomComMinisomBiblioteca.py) | SOM com MiniSom                                                           |
| [GO0806–07](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class08/)                                 | SOM em negócios e detecção de anomalias                                   |
| [GO0809–14](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class08/)                                 | Atividade: dataset de vinhos — treino, visualizações, análise de clusters |
| [GO0815](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class08/GO0815-Hiperparametros.py)           | Hiperparâmetros do SOM                                                    |

---

### Aula 09 — Fundamentos de Redes Neurais `class09/` · 30 scripts

Perceptron, backpropagation, funções de ativação, regularização e inicialização de pesos.

| Código                                                                                                           | Descrição                                                                 |
| ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| [GO0901–03](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class09/)                                    | Uso moderno, resumo e treinamento de redes neurais                        |
| [GO0905](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class09/GO0905-ChecklistDeDebugging.py)         | Checklist de debugging de redes neurais                                   |
| [GO0906–08](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class09/)                                    | MNIST Parte 2: arquitetura, treinamento, análise de erros                 |
| [GO0909](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class09/GO0909-3DropoutPadraoEmDeepLearning.py) | Dropout como regularização padrão                                         |
| [GO0910](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class09/GO0910-Adam.py)                         | Otimizador Adam                                                           |
| [GO0914](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class09/GO0914-AlgoritmoTreinamentoCompleto.py) | Algoritmo de treinamento completo                                         |
| [GO0925–29](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class09/)                                    | Inicialização de pesos, batch norm, verificação de gradientes e ativações |
| [GO0930–32](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class09/)                                    | MLP com scikit-learn e TensorFlow                                         |

---

### Aula 10 — MLP — Redes Multicamadas `class10/` · 53 scripts

Redes densas profundas com Keras/TensorFlow: callbacks, regularização, salvamento e Tensorboard.

| Código                                                                                                   | Descrição                                   |
| -------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| [GO1001–08](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class10/)                            | MLP com TensorFlow/Keras                    |
| [GO1009](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class10/GO1009-ClassificacaoBinaria.py) | Classificação binária completa              |
| [GO1012](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class10/GO1012-Callbacks.py)            | Callbacks (EarlyStopping, ModelCheckpoint)  |
| [GO1013](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class10/GO1013-History.py)              | History e visualização de treinamento       |
| [GO1021–26](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class10/)                            | Salvar modelo completo, pesos, JSON         |
| [GO1034–35](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class10/)                            | Data pipeline (load, preprocess)            |
| [GO1037–39](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class10/)                            | Tensorboard e logging                       |
| [GO1044](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class10/GO1044-CustomCallback.py)       | Custom Callback                             |
| [GO1050–51](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class10/)                            | Learning rate scheduling e cosine annealing |
| [GO1054](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class10/GO1054-GetSaliencyMap.py)       | Saliency map                                |

---

### Aula 11 — Lógica Fuzzy `class11/` · 21 scripts

Conjuntos fuzzy, funções de pertinência, sistemas de inferência e controle fuzzy.

| Código                                                                                                                   | Descrição                                                   |
| ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| [GO1101–07](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class11/)                                            | Implementações de conjuntos e operadores fuzzy              |
| [GO1108](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class11/GO1108-SistemaDeControle.py)                    | Sistema de controle fuzzy                                   |
| [GO1109](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class11/GO1109-OperadoresCustomizados.py)               | Operadores customizados                                     |
| [GO1112](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class11/GO1112-ProjetoControladorDeTemperaturaParte.py) | Projeto: controlador de temperatura                         |
| [GO1113–16](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class11/)                                            | Implementação conceitual, treinamento híbrido, convergência |
| [GO1121](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class11/GO1121-Simpful.py)                              | Simpful — biblioteca de lógica fuzzy                        |

---

### Aula 12 — CNNs — Redes Convolucionais `class12/` · 62 scripts

Arquiteturas CNN clássicas e modernas, transfer learning, visualização e técnicas avançadas.

| Código                                                                                                                            | Descrição                                                 |
| --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| [GO1201](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1201-CnnMnistPassoAPasso.py)                           | CNN no MNIST (passo a passo)                              |
| [GO1202](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1202-DataAugmentationMelhorandoGeneralizacao.py)       | Data augmentation                                         |
| [GO1203](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1203-11cTransferLearningResnet50.py)                   | Transfer learning com ResNet50                            |
| [GO1204–05](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class12/)                                                     | Visualização de filtros, feature maps e GradCAM           |
| [GO1208A](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1208A-VisionTransformersViT.py)                       | Vision Transformers (ViT)                                 |
| [GO1208B](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1208B-ViTPretreinadoHuggingFace.py)                   | ViT pré-treinado com HuggingFace                          |
| [GO1209](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1209-35gNeuralStyleTransferArteCom.py)                 | Neural Style Transfer                                     |
| [GO1210](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1210-35hCnnFeatureVisualizationEntendendoFiltros.py)   | CNN feature visualization                                 |
| [GO1211](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1211-35iMixedPrecisionTrainingAceleracaoCom.py)        | Mixed Precision Training                                  |
| [GO1212](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1212-35jCnnModelPruningCompressaoInteligente.py)       | Model pruning                                             |
| [GO1213](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1213-35kSingleShotDetectorSsdObject.py)                | SSD — Single Shot Detector                                |
| [GO1214](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1214-35lImageSuperresolutionAumentarResolucao.py)      | Super-resolução de imagens                                |
| [GO1215](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1215-35mCycleganStyleTransferSemPares.py)              | CycleGAN                                                  |
| [GO1216](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1216-35nSemanticSegmentationUnetSegmentacaoPixel.py)   | Segmentação semântica com U-Net                           |
| [GO1217](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1217-35oAdversarialRobustnessDefesaContraAtaques.py)   | Robustez adversarial                                      |
| [GO1218](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1218-35pAutoaugmentDataAugmentationAutomatico.py)      | AutoAugment                                               |
| [GO1219](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1219-35qKnowledgeDistillationComprimirConhecimento.py) | Knowledge Distillation                                    |
| [GO1220](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1220-35rEfficientnetNeuralArchitectureSearch.py)       | EfficientNet / NAS                                        |
| [GO1221](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1221-35sMaskRcnnInstanceSegmentation.py)               | Mask R-CNN — Instance Segmentation                        |
| [GO1222](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class12/GO1222-35t3dCnnVideoClassification.py)                   | 3D CNN para classificação de vídeo                        |
| [GO1236–41](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class12/)                                                     | AlexNet, VGG16, ResNet (residual block), arquiteturas CNN |

---

### Aula 13 — Visão Computacional & Reconhecimento de Padrões `class13a/` · 44 scripts

Transfer learning avançado, YOLO, detecção de objetos, métricas de CV e deploy embarcado.

| Código                                                                                                      | Descrição                                                 |
| ----------------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| [GO1301–03](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class13a/)                              | Transfer learning: carregar, descongelar, fine-tune       |
| [GO1304–10](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class13a/)                              | YOLO com Ultralytics: inferência, tracking, vídeo, câmera |
| [GO1312](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class13a/GO1312-LearningRateScheduling.py) | Learning rate scheduling                                  |
| [GO1313](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class13a/GO1313-FastapiUltralytics.py)     | Deploy de YOLO com FastAPI                                |
| [GO1317–19](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class13a/)                              | IoU: GIoU, DIoU, CIoU                                     |
| [GO1323–24](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class13a/)                              | Cálculo de AP (11 pontos e todos os pontos)               |
| [GO1331](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class13a/GO1331-CocoToYoloSeg.py)          | Converter anotações COCO → YOLO                           |
| [GO1333–35](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class13a/)                              | Análise de área, remoção de background, sobreposição      |
| [GO1337–39](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class13a/)                              | Export: TFLite, ONNX, TensorRT                            |

---

### Aula 14 — RNN / LSTM `class14/` · 41 scripts

Redes recorrentes e LSTM para séries temporais, NLP e geração de sequências.

| Código                                                                                                                        | Descrição                                                                       |
| ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| [GO1401](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class14/GO1401-5SimpleRnnEmKeras.py)                         | SimpleRNN no Keras                                                              |
| [GO1402](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class14/GO1402-12LstmEmKeras.py)                             | LSTM no Keras                                                                   |
| [GO1403](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class14/GO1403-PrevisaoDeConsumoDeEnergiaComLSTM.py)         | Previsão de consumo de energia com LSTM                                         |
| [GO1404–05](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class14/)                                                 | Text classification e análise de sentimento (IMDB) com LSTM                     |
| [GO1406](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class14/GO1406-LstmBidirecionalContextoFuturoEPassado.py)    | LSTM bidirecional                                                               |
| [GO1409](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class14/GO1409-15c24SequencetosequenceTra.py)                | Seq2Seq (sequence-to-sequence)                                                  |
| [GO1410](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class14/GO1410-GeracaoDeTextoComLstmCharacterlevel.py)       | Geração de texto com LSTM (character-level)                                     |
| [GO1414–15](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class14/)                                                 | LSTM Encoder-Decoder e com atenção                                              |
| [GO1417](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class14/GO1417-36bOpcionalPrevisaoDeSeriesTemporais.py)      | Previsão de séries temporais avançada                                           |
| [GO1420](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class14/GO1420-36eOpcionalEncoderdecoderLstmParaTraducao.py) | Encoder-Decoder para tradução                                                   |
| [GO1422](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class14/GO1422-36gMusicGenerationComLstmComposicao.py)       | Geração de música com LSTM                                                      |
| [GO1425–30](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class14/)                                                 | NER, QA, extração de speech features, sumarização, tradução, sistemas dialogais |
| [GO1433–35](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class14/)                                                 | Análise médica de séries temporais, visualização de atenção, trading bot        |

---

### Aula 15 — Introdução ao NLP `class15/` · 42 scripts

Processamento de linguagem natural clássico e com deep learning.

| Código                                                                                                                  | Descrição                                                   |
| ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| [GO1501–02](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class15/)                                           | Tokenização e normalização de texto                         |
| [GO1503](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class15/GO1503-4aCompleteTextNormalizationPipeline.py) | Pipeline completo de normalização                           |
| [GO1504–05](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class15/)                                           | Stemming e lemmatização                                     |
| [GO1506–07](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class15/)                                           | Bag of Words (BoW)                                          |
| [GO1509](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class15/GO1509-8Ngrams.py)                             | N-gramas                                                    |
| [GO1514–15](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class15/)                                           | Word2Vec                                                    |
| [GO1516–17](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class15/)                                           | FastText e análise de subpalavras                           |
| [GO1518](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class15/GO1518-15SimilaridadeEntrePalavras.py)         | Similaridade entre palavras                                 |
| [GO1520](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class15/GO1520-16EmbeddingsEmDeepLearning.py)          | Embeddings em deep learning                                 |
| [GO1521](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class15/GO1521-17Projeto1AnaliseDeSentimento.py)       | Projeto: análise de sentimento                              |
| [GO1522–24](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class15/)                                           | POS tagging e NER (completo com spaCy)                      |
| [GO1525–26](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class15/)                                           | Classificação de texto com deep learning e CNN              |
| [GO1527–28](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class15/)                                           | Topic modeling com LDA                                      |
| [GO1529–31](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class15/)                                           | Sumarização, Question Answering                             |
| [GO1532](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class15/GO1532-TraducaoNeuralMarianMT.py)              | Tradução neural com MarianMT                                |
| [GO1534–41](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class15/)                                           | Classificador de notícias, dataset IMDB, sistemas completos |

---

### Aula 16 — Transformers & LLMs `class16/` · 44 scripts

Arquitetura Transformer, BERT, GPT-2, T5, fine-tuning, LoRA, quantização e segurança.

| Código                                                                                                                | Descrição                                                                                                                                                               |
| --------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [GO1601](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class16/GO1601-6aAttentionVisualizationAnalysis.py)  | Visualização e análise de attention                                                                                                                                     |
| [GO1602](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class16/GO1602-13aT5MultitaskLearning.py)            | T5 — Multitask Learning                                                                                                                                                 |
| [GO1603](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class16/GO1603-13bChainofthoughtPrompting.py)        | Chain-of-Thought Prompting                                                                                                                                              |
| [GO1604](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class16/GO1604-14aBertFinetuningEndtoend.py)         | Fine-tuning end-to-end com BERT                                                                                                                                         |
| [GO1605](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class16/GO1605-15aLlmBenchmarkingModelSelection.py)  | Benchmarking e seleção de LLMs                                                                                                                                          |
| [GO1606](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class16/GO1606-16aMultimodalLlmsVisionLanguage.py)   | LLMs multimodais — visão + linguagem                                                                                                                                    |
| [GO1607](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class16/GO1607-17HuggingFaceTransformers.py)         | HuggingFace Transformers — guia prático                                                                                                                                 |
| [GO1609](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class16/GO1609-18aAdvancedTokenizationComparison.py) | Comparação avançada de tokenizadores                                                                                                                                    |
| [GO1611](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class16/GO1611-21ProjetoChatbotComGpt2.py)           | Projeto: Chatbot com GPT-2                                                                                                                                              |
| [GO1612](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class16/GO1612-21aZeroshotFewshotClassification.py)  | Zero-shot e few-shot classification                                                                                                                                     |
| [GO1613](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class16/GO1613-21bModelDistillation.py)              | Model distillation                                                                                                                                                      |
| [GO1614](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class16/GO1614-21cAdvancedSamplingStrategies.py)     | Estratégias avançadas de sampling                                                                                                                                       |
| [GO1615](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class16/GO1615-22LoraLowrankAdaptation.py)           | LoRA — Low-Rank Adaptation                                                                                                                                              |
| [GO1616–17](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class16/)                                         | Quantização básica e técnicas avançadas                                                                                                                                 |
| [GO1619–20](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class16/)                                         | Embeddings e busca semântica                                                                                                                                            |
| [GO1621](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class16/GO1621-29aLongContextHandlingStrategies.py)  | Estratégias para contextos longos                                                                                                                                       |
| [GO1234A](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class16/GO1234A-ScalingLawsLLMs.py)                 | Scaling Laws para LLMs                                                                                                                                                  |
| [GO1622](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class16/GO1622-30aLlmCostOptimizationCalculator.py)  | Calculadora de custo de LLMs                                                                                                                                            |
| [GO1623–25](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class16/)                                         | Detecção de alucinação, viés e defesa contra prompt injection                                                                                                           |
| [GO1627–36](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class16/)                                         | Exercícios avançados: self-attention, positional encoding, BERT vs GPT, fine-tuning, QA, geração de texto, sentence embeddings, MLM, LoRA, análise de attention weights |

---

### Aula 17 — LLMs Locais, Ollama & RAG `class17/` · 39 scripts

Execução local de LLMs, pipelines RAG, vector stores e aplicações conversacionais.

| Código                                                                                                                   | Descrição                                                   |
| ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------- |
| [GO1701](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class17/GO1701-10VectorStores.py)                       | Vector stores                                               |
| [GO1704–05](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class17/)                                            | Metadata filtering, query routing e estratégias de chunking |
| [GO1706](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class17/GO1706-19ProjetoChatbotRagComOllama.py)         | Projeto: Chatbot RAG com Ollama                             |
| [GO1707](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class17/GO1707-19aProductionMonitoringObservability.py) | Monitoramento e observabilidade em produção                 |
| [GO1708–10](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class17/)                                            | Query expansion, GraphRAG, otimização de custo              |
| [GO1712](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class17/GO1712-24AgentsComRag.py)                       | Agents com RAG                                              |
| [GO1713–15](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class17/)                                            | Streamlit UI, app RAG production-ready, QA e teste          |
| [GO1716](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class17/GO1716-29aPromptInjectionProtection.py)         | Proteção contra prompt injection                            |
| [GO1717](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class17/GO1717-30aHybridSearchBm25VectorSearch.py)      | Busca híbrida: BM25 + vector search                         |
| [GO1718](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class17/GO1718-30bRagEvaluationMetrics.py)              | Métricas de avaliação de RAG                                |
| [GO1719](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class17/GO1719-30cStreamingResponses.py)                | Streaming de respostas                                      |
| [GO1721](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class17/GO1721-30dConversationalMemorySystems.py)       | Sistemas de memória conversacional                          |
| [GO1723](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class17/GO1723-34aVectorStoreComparisonFaissVs.py)      | Comparação de vector stores: FAISS vs alternativas          |
| [GO1345A](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class17/GO1345A-LoraQLoraFineTuning.py)                | QLoRA fine-tuning com quantização 4-bit                     |
| [GO1345B](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class17/GO1345B-UsarModeloLoRAFineTunado.py)           | Usar modelo LoRA fine-tunado                                |

---

### Aula 18 — Reinforcement Learning `class18/` · 45 scripts

Q-Learning, DQN e algoritmos de RL moderno com Gymnasium.

| Código                                                                                                                        | Descrição                                                                                    |
| ----------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| [GO1806](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class18/GO1806-9aGridWorldQlearning.py)                      | Grid World com Q-Learning                                                                    |
| [GO1809](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class18/GO1809-15DqnSetup.py)                                | DQN — setup                                                                                  |
| [GO1815](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class18/GO1815-24AmbientesGymnasium.py)                      | Ambientes Gymnasium                                                                          |
| [GO1817](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class18/GO1817-26ProjetoDqnCartpoleSetup.py)                 | Projeto DQN no CartPole                                                                      |
| [GO1818](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class18/GO1818-30DosagemDinamica.py)                         | Manipulação robótica, AlphaZero simplificado, data center, gestão de portfólio, recomendação |
| [GO1819](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class18/GO1819-Exercicio2DqnNoCartpole.py)                   | DQN no CartPole (exercício)                                                                  |
| [GO1820](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class18/GO1820-Exercicio3CompararAlgoritmosNoLunarlander.py) | Comparação de algoritmos no LunarLander                                                      |
| [GO1822–39](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class18/)                                                 | Snippets: discretização, reward shaping, Q-tables, políticas                                 |

---

### Aula 19 — Algoritmos Genéticos & Metaheurísticas `class19/` · 33 scripts

Algoritmos evolutivos, bio-inspirados e otimização combinatória.

| Código                                                                                                                    | Descrição                                                                     |
| ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| [GO1906](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1906-6PopulacaoInicial.py)                     | Geração de população inicial                                                  |
| [GO1907](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1907-9ProjetoMaximizarFuncaoMatematica.py)     | Projeto: maximizar função matemática com AG                                   |
| [GO1908](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1908-10SelecaoTorneioERoleta.py)               | Seleção por torneio e roleta                                                  |
| [GO1909](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1909-14aAvancadoFuncaoDeRosenbrock.py)         | Função de Rosenbrock                                                          |
| [GO1918](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1918-Slides1822ProjetoTspTravelingSalesman.py) | Projeto TSP — Travelling Salesman Problem                                     |
| [GO1919](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1919-Slides1822SeleECrossover.py)              | Neuroevolução de redes neurais                                                |
| [GO1920](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1920-DifferentialEvolutionDoZero.py)           | Differential Evolution do zero                                                |
| [GO1921](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1921-ParticleSwarmOptimizationCompleta.py)     | Particle Swarm Optimization (PSO) completo                                    |
| [GO1922](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1922-AntColonyOptimizationCompletaTSP.py)      | Ant Colony Optimization (ACO) para TSP                                        |
| [GO1920A](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1920A-DEHyperparameterTuningAutoML.py)        | DE para hyperparameter tuning AutoML                                          |
| [GO1920B](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1920B-DEOtimizacaoCircuitosEletricos.py)      | DE para otimização de circuitos elétricos                                     |
| [GO1920C](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1920C-DEOtimizacaoEstrutursMecanicas.py)      | DE para otimização de estruturas mecânicas                                    |
| [GO1921A](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1921A-PSOTreinarRedesNeurais.py)              | PSO para treinar redes neurais                                                |
| [GO1921B](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1921B-PSOOtimizacaoSistemasEnergeticos.py)    | PSO para otimização de sistemas energéticos                                   |
| [GO1922A](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1922A-ACORoteamentoLogistica.py)              | ACO para roteamento de logística                                              |
| [GO1922B](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1922B-ACORoteamentoRedesTelecom.py)           | ACO para roteamento de redes telecom                                          |
| [GO1923](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1923-CMAESBiblioteca.py)                       | CMA-ES com biblioteca                                                         |
| [GO1923A](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1923A-OpenAIEvolutionStrategiesRL.py)         | OpenAI Evolution Strategies para RL                                           |
| [GO1923B](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1923B-CMAESControleRobotico.py)               | CMA-ES para controle robótico                                                 |
| [GO1924](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1924-GeneticProgrammingSymbolicRegression.py)  | Genetic Programming — regressão simbólica                                     |
| [GO1924A](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1924A-NEATJogarFlappyBird.py)                 | NEAT para jogar Flappy Bird                                                   |
| [GO1925](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1925-BenchmarkComparativoAlgoritmos.py)        | Benchmark comparativo de algoritmos                                           |
| [GO1925A](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1925A-NSGAIIMultiObjetivo.py)                 | NSGA-II multi-objetivo                                                        |
| [GO1925B](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1925B-NSGAIIPortfolioFinanceiro.py)           | NSGA-II para portfolio financeiro                                             |
| [GO1926–28](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class19/)                                             | Exercícios: função multimodal, problema da mochila, tuning de hiperparâmetros |
| [GO1929–30](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class19/)                                             | TSP com AG e Simulated Annealing                                              |
| [GO1931](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class19/GO1931-OtimizacaoMultiobjetivoMinimizeF1EF2.py)  | Otimização multiobjetivo                                                      |

---

### Aula 20 — MLOps & Deploy `class20/` · 20 scripts

Ciclo de vida de modelos em produção: rastreamento, API, monitoramento e CI/CD.

| Código                                                                                                                   | Descrição                                              |
| ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------ |
| [GO2001–04](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class20/)                                            | MLflow: tracking, UI, autolog, model registry          |
| [GO2005–07](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class20/)                                            | FastAPI: projeto Iris API, salvar/rodar modelo, testes |
| [GO2008–09](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class20/)                                            | Azure Machine Learning — treinamento e inferência      |
| [GO2013](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class20/GO2013-33PipelineDeRetreinoCompleto.py)         | Pipeline de retreino completo                          |
| [GO2015](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class20/GO2015-34BoasPraticasMlops.py)                  | Boas práticas de MLOps                                 |
| [GO2017](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class20/GO2017-26HyperparameterTuningAvancadoDaAula.py) | Hyperparameter tuning avançado                         |
| [GO2018](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class20/GO2018-27OptunaFrameworkModernoDeOtimizacao.py) | Optuna — framework moderno de otimização               |
| [GO2019](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class20/GO2019-28LearningRateSchedulingDaAula.py)       | Learning rate scheduling                               |
| [GO2020](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class21/GO2020-15CLIMETexto.py)                         | Model monitoring e drift detection                     |
| [GO2021–22](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class20/)                                            | Testes automatizados com pytest                        |
| [GO2023](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class20/GO2023-FlaskPrometheus_Flask_Exporter.py)       | Flask + Prometheus para métricas                       |

---

### Aula 21 — IA Generativa & Futuro `class21/` · 26 scripts

GANs, VAEs, Stable Diffusion, XAI, Small Language Models e tópicos avançados de ética e segurança em IA.

| Código                                                                                                                       | Descrição                                                                               |
| ---------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| [GO2020](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class21/GO2020-15CLIMETexto.py)                             | SHAP na prática; LIME para texto                                                        |
| [GO2101](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class21/GO2101-GanParaMnist.py)                             | GAN para MNIST                                                                          |
| [GO2102](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class21/GO2102-715GanCompletoComMonitoram.py)               | GAN completo com monitoramento                                                          |
| [GO2103](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class21/GO2103-StableDiffusionComDiffusersHuggingFace.py)   | Stable Diffusion com Diffusers (HuggingFace)                                            |
| [GO2104](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class21/GO2104-VaeVariationalAutoencoderParaMnist.py)       | VAE — Variational Autoencoder para MNIST                                                |
| [GO2105](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class21/GO2105-StableDiffusionGeracaoAvancadaDeImagens.py)  | Stable Diffusion — geração avançada de imagens                                          |
| [GO2106](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class21/GO2106-VaeCustomFinetuningParaDatasetEspecifico.py) | VAE fine-tuning para dataset específico                                                 |
| [GO2107–08](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class21/)                                                | XAI: GradCAM, SHAP e LIME                                                               |
| [GO2109](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class21/GO2109-FairnessEMitigacaoDeVies.py)                 | Fairness e mitigação de viés                                                            |
| [GO2110](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class21/GO2110-1816AdversarialAttacksEDef.py)               | Ataques adversariais e defesas                                                          |
| [GO2111](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class21/GO2111-1827DifferentialPrivacy.py)                  | Differential Privacy                                                                    |
| [GO2112–16](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class21/)                                                | Exercícios avançados: GAN do zero, VAE, LIME, Stable Diffusion, pipeline MLOps completo |
| [GO1456A](https://github.com/giovanebarcelos/ai-sandbox/blob/main/class21/GO1456A-SmallLanguageModelsSlms.py)                | Small Language Models (SLMs) — Phi-3 Mini                                               |
| [GO2121–23](https://github.com/giovanebarcelos/ai-sandbox/tree/main/class21/)                                                | StarCoder, LLaMA, Mistral em execução local                                             |

---

## Nomenclatura dos Arquivos

Os arquivos seguem o padrão `GO<aula><sequência>-<NomeDescritivo>.py`:

```
GO0501-KnearestNeighborsDoZero.py
│  │  │
│  │  └─ Número sequencial dentro da aula
│  └──── Número da aula (01–21)
└─────── Prefixo do projeto
```

---

## Como Executar

A maioria dos scripts usa apenas bibliotecas padrão do ecossistema Python. Para instalar as dependências mais comuns:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow keras torch \
            xgboost imbalanced-learn opencv-python ultralytics \
            transformers datasets sentence-transformers faiss-cpu \
            mlflow fastapi uvicorn optuna simpful minisom
```

Para scripts que utilizam Ollama (aula 17):

```bash
# Instale o Ollama: https://ollama.com
ollama pull llama3
```

Executar um script individualmente:

```bash
cd class05
python3 GO0501-KnearestNeighborsDoZero.py
```

---

## Sumário

| Aula      | Tema                                            | Scripts |
| --------- | ----------------------------------------------- | ------: |
| 01        | Introdução à IA & Ética                         |      22 |
| 02        | Representação do Conhecimento                   |      22 |
| 03        | Resolução de Problemas & Busca                  |      15 |
| 04        | Introdução ao Machine Learning                  |      32 |
| 05        | Algoritmos de Classificação                     |      43 |
| 06        | Regressão & Validação                           |      36 |
| 07        | Clustering                                      |      14 |
| 08        | SOM — Mapas Auto-Organizáveis                   |      15 |
| 09        | Fundamentos de Redes Neurais                    |      30 |
| 10        | MLP — Redes Multicamadas                        |      53 |
| 11        | Lógica Fuzzy                                    |      21 |
| 12        | CNNs — Redes Convolucionais                     |      62 |
| 13        | Visão Computacional & Reconhecimento de Padrões |      44 |
| 14        | RNN / LSTM                                      |      41 |
| 15        | Introdução ao NLP                               |      42 |
| 16        | Transformers & LLMs                             |      44 |
| 17        | LLMs Locais, Ollama & RAG                       |      39 |
| 18        | Reinforcement Learning                          |      45 |
| 19        | Algoritmos Genéticos & Metaheurísticas          |      44 |
| 20        | MLOps & Deploy                                  |      20 |
| 21        | IA Generativa & Futuro                          |      26 |
| **Total** |                                                 | **711** |

---

## Licença

Material produzido para fins acadêmicos na disciplina de Inteligência Artificial — Pós-Graduação FAPA, 2026.
