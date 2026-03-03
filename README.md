# ai-sandbox

Repositório de código-fonte da disciplina **Inteligência Artificial** — Pós-Graduação FAPA (2026).

Contém **691 scripts Python** organizados por aula, cobrindo desde os fundamentos de IA até tópicos avançados como LLMs, IA generativa e MLOps.

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

| Categoria | Bibliotecas |
|-----------|-------------|
| Machine Learning | scikit-learn, XGBoost, imbalanced-learn |
| Deep Learning | TensorFlow / Keras, PyTorch |
| Visão Computacional | OpenCV, Ultralytics (YOLO), Pillow |
| NLP | NLTK, spaCy, Gensim, HuggingFace Transformers |
| LLMs / RAG | Ollama, LangChain, FAISS, sentence-transformers |
| Otimização | DEAP, scipy, CMA-ES |
| MLOps | MLflow, FastAPI, Optuna, Prometheus |
| Visualização | Matplotlib, Seaborn, Tensorboard |
| Fuzzy | scikit-fuzzy, simpful |

---

## Conteúdo por Aula

### Aula 01 — Introdução à IA & Ética `class01/` · 22 scripts

Panorama da IA moderna, comparação de abordagens (simbólica vs conexionista) e dimensões éticas.

| Código | Descrição |
|--------|-----------|
| GO0101 | Simulador do Teste de Turing |
| GO0102 | Análise de viés em sistemas de IA |
| GO0103 | Benchmarks de Inteligência Artificial |
| GO0104 | Classificador simples (regras vs ML) |
| GO0105–06 | Programação tradicional por regras vs Machine Learning |
| GO0107 | Modelo pré-treinado MobileNetV2 |
| GO0108 | Sistema de recomendação em 30 minutos |
| GO0113 | Experimento: capacidades e limitações de LLMs |
| GO0114 | Agente autônomo — robô aspirador |
| GO0115 | Simulador de dilemas éticos em IA |
| GO0116 | Comparação IA simbólica vs conexionista |
| GO0117 | Avaliador interativo de AGI |
| GO0118 | XAI — técnicas de interpretabilidade |
| GO0119 | Impacto da IA no trabalho (simulador) |
| GO0122 | Gerador de roadmap pessoal de carreira |

---

### Aula 02 — Representação do Conhecimento `class02/` · 21 scripts

Sistemas especialistas, motores de inferência, grafos de conhecimento e raciocínio incerto.

| Código | Descrição |
|--------|-----------|
| GO0201 | Fatores de certeza (CF) |
| GO0203 | Sistema especialista de recomendação de filmes |
| GO0204–05 | Motor de inferência e execução |
| GO0208 | Grafo de conhecimento médico |
| GO0210 | Motor de inferência com forward chaining |
| GO0211 | Prolog simplificado em Python |
| GO0212 | Lógica fuzzy para decisões |
| GO0213 | Raciocínio estruturado |
| GO0214 | CSP — Constraint Satisfaction Problems |
| GO0215 | STRIPS Planning |
| GO0216 | Redes semânticas |
| GO0217 | Raciocínio bayesiano |
| GO0218 | Sistemas de produção |

---

### Aula 03 — Resolução de Problemas & Busca `class03/` · 14 scripts

Algoritmos de busca cega e heurística para resolução de problemas em grafos e labirintos.

| Código | Descrição |
|--------|-----------|
| GO0301 | Representação de estados de busca |
| GO0306 | Heurísticas comuns (Manhattan, Euclidiana) |
| GO0307 | Visualização de labirinto e solução |
| GO0311 | Comparação empírica BFS vs A* |
| GO0312 | Busca de Dijkstra em grafo ponderado |
| GO0313 | Puzzle das 8 peças com A* |
| GO0314 | Busca bidirecional |

---

### Aula 04 — Introdução ao Machine Learning `class04/` · 32 scripts

Fundamentos do ML supervisionado com scikit-learn: pipeline, validação e diagnóstico.

| Código | Descrição |
|--------|-----------|
| GO0403 | Pipeline completo — flores Iris |
| GO0404 | Learning curves — diagnóstico de overfitting/underfitting |
| GO0405 | Validação cruzada com scikit-learn |
| GO0406 | Otimização de hiperparâmetros com GridSearch |
| GO0407 | Pipeline completo de Machine Learning |
| GO0412–20 | Exercícios: EDA, preprocessamento, feature engineering, data leakage |
| GO0418–20 | Exercícios avançados: pipeline automatizado, curvas de aprendizado |
| GO0422 | Detecção de spam (abordagem tradicional) |

---

### Aula 05 — Algoritmos de Classificação `class05/` · 43 scripts

KNN, árvores de decisão, Naive Bayes, métricas de avaliação, ensemble e desbalanceamento.

| Código | Descrição |
|--------|-----------|
| GO0501 | KNN do zero |
| GO0502 | Árvore de decisão com scikit-learn |
| GO0503 | Naive Bayes com scikit-learn |
| GO0504 | Matriz de confusão e análise |
| GO0505 | ROC Curve e AUC |
| GO0506 | Ajuste de threshold de decisão |
| GO0507 | Métricas para problemas multiclasse |
| GO0508–12 | Prática MNIST: preparação, treino, análise de erros, otimização |
| GO0509 | Diagnóstico de diabetes com Ensemble |
| GO0510 | Classificação de imagens com Transfer Learning |
| GO0521–27 | Random Forest, XGBoost, SMOTE, Imbalanced-learn |
| GO0540 | Dataset de crédito — classificação real |

---

### Aula 06 — Regressão & Validação `class06/` · 36 scripts

Regressão linear, polinomial e regularizada; feature engineering e análise de resíduos.

| Código | Descrição |
|--------|-----------|
| GO0601 | Regressão linear e múltipla |
| GO0602 | Regressão polinomial |
| GO0603 | Validação cruzada para regressão |
| GO0604 | Regularização Ridge (L2) |
| GO0605 | Regularização Lasso (L1) |
| GO0606 | ElasticNet (L1 + L2) |
| GO0607–08 | Feature engineering e análise de resíduos |
| GO0609–14 | Atividade prática: previsão de preços, EDA, treinamento, validação |
| GO0618–21 | Coeficientes padronizados, importância de features, learning curves |

---

### Aula 07 — Clustering `class07/` · 14 scripts

Aprendizado não supervisionado: K-Means, DBSCAN e clustering hierárquico.

| Código | Descrição |
|--------|-----------|
| GO0701 | Visualização do processo de clustering |
| GO0702 | Gráfico do cotovelo (elbow method) |
| GO0703 | DBSCAN com scikit-learn |
| GO0704 | Dendrograma de corte |
| GO0706 | Segmentação de clientes para marketing |
| GO0707–09 | Pipeline: normalização, análise de clusters, estratégias de marketing |

---

### Aula 08 — SOM — Mapas Auto-Organizáveis `class08/` · 15 scripts

Self-Organizing Maps com MiniSom para clustering, visualização e detecção de anomalias.

| Código | Descrição |
|--------|-----------|
| GO0801 | Fases do treinamento SOM |
| GO0802 | Uso de cores/canais RGB |
| GO0805 | SOM com MiniSom |
| GO0806–07 | SOM em negócios e detecção de anomalias |
| GO0809–14 | Atividade: dataset de vinhos — treino, visualizações, análise de clusters |
| GO0815 | Hiperparâmetros do SOM |

---

### Aula 09 — Fundamentos de Redes Neurais `class09/` · 30 scripts

Perceptron, backpropagation, funções de ativação, regularização e inicialização de pesos.

| Código | Descrição |
|--------|-----------|
| GO0901–03 | Uso moderno, resumo e treinamento de redes neurais |
| GO0905 | Checklist de debugging de redes neurais |
| GO0906–08 | MNIST Parte 2: arquitetura, treinamento, análise de erros |
| GO0909 | Dropout como regularização padrão |
| GO0910 | Otimizador Adam |
| GO0914 | Algoritmo de treinamento completo |
| GO0925–29 | Inicialização de pesos, batch norm, verificação de gradientes e ativações |
| GO0930–32 | MLP com scikit-learn e TensorFlow |

---

### Aula 10 — MLP — Redes Multicamadas `class10/` · 53 scripts

Redes densas profundas com Keras/TensorFlow: callbacks, regularização, salvamento e Tensorboard.

| Código | Descrição |
|--------|-----------|
| GO1001–08 | MLP com TensorFlow/Keras |
| GO1009 | Classificação binária completa |
| GO1012 | Callbacks (EarlyStopping, ModelCheckpoint) |
| GO1013 | History e visualização de treinamento |
| GO1021–26 | Salvar modelo completo, pesos, JSON |
| GO1034–35 | Data pipeline (load, preprocess) |
| GO1037–39 | Tensorboard e logging |
| GO1044 | Custom Callback |
| GO1050–51 | Learning rate scheduling e cosine annealing |
| GO1054 | Saliency map |

---

### Aula 11 — Lógica Fuzzy `class11/` · 21 scripts

Conjuntos fuzzy, funções de pertinência, sistemas de inferência e controle fuzzy.

| Código | Descrição |
|--------|-----------|
| GO1101–07 | Implementações de conjuntos e operadores fuzzy |
| GO1108 | Sistema de controle fuzzy |
| GO1109 | Operadores customizados |
| GO1112 | Projeto: controlador de temperatura |
| GO1113–16 | Implementação conceitual, treinamento híbrido, convergência |
| GO1121 | Simpful — biblioteca de lógica fuzzy |

---

### Aula 12 — CNNs — Redes Convolucionais `class12/` · 60 scripts

Arquiteturas CNN clássicas e modernas, transfer learning, visualização e técnicas avançadas.

| Código | Descrição |
|--------|-----------|
| GO1201 | CNN no MNIST (passo a passo) |
| GO1202 | Data augmentation |
| GO1203 | Transfer learning com ResNet50 |
| GO1204–05 | Visualização de filtros, feature maps e GradCAM |
| GO1209 | Neural Style Transfer |
| GO1210 | CNN feature visualization |
| GO1211 | Mixed Precision Training |
| GO1212 | Model pruning |
| GO1213 | SSD — Single Shot Detector |
| GO1214 | Super-resolução de imagens |
| GO1215 | CycleGAN |
| GO1216 | Segmentação semântica com U-Net |
| GO1217 | Robustez adversarial |
| GO1218 | AutoAugment |
| GO1219 | Knowledge Distillation |
| GO1220 | EfficientNet / NAS |
| GO1221 | Mask R-CNN — Instance Segmentation |
| GO1222 | 3D CNN para classificação de vídeo |
| GO1236–41 | AlexNet, VGG16, ResNet (residual block), arquiteturas CNN |

---

### Aula 13 — Visão Computacional & Reconhecimento de Padrões `class13a/` · 44 scripts

Transfer learning avançado, YOLO, detecção de objetos, métricas de CV e deploy embarcado.

| Código | Descrição |
|--------|-----------|
| GO1301–03 | Transfer learning: carregar, descongelar, fine-tune |
| GO1304–10 | YOLO com Ultralytics: inferência, tracking, vídeo, câmera |
| GO1312 | Learning rate scheduling |
| GO1313 | Deploy de YOLO com FastAPI |
| GO1317–19 | IoU: GIoU, DIoU, CIoU |
| GO1323–24 | Cálculo de AP (11 pontos e todos os pontos) |
| GO1331 | Converter anotações COCO → YOLO |
| GO1333–35 | Análise de área, remoção de background, sobreposição |
| GO1337–39 | Export: TFLite, ONNX, TensorRT |

---

### Aula 14 — RNN / LSTM `class14/` · 41 scripts

Redes recorrentes e LSTM para séries temporais, NLP e geração de sequências.

| Código | Descrição |
|--------|-----------|
| GO1401 | SimpleRNN no Keras |
| GO1402 | LSTM no Keras |
| GO1403 | Previsão de consumo de energia com LSTM |
| GO1404–05 | Text classification e análise de sentimento (IMDB) com LSTM |
| GO1406 | LSTM bidirecional |
| GO1409 | Seq2Seq (sequence-to-sequence) |
| GO1410 | Geração de texto com LSTM (character-level) |
| GO1414–15 | LSTM Encoder-Decoder e com atenção |
| GO1417 | Previsão de séries temporais avançada |
| GO1420 | Encoder-Decoder para tradução |
| GO1422 | Geração de música com LSTM |
| GO1425–30 | NER, QA, extração de speech features, sumarização, tradução, sistemas dialogais |
| GO1433–35 | Análise médica de séries temporais, visualização de atenção, trading bot |

---

### Aula 15 — Introdução ao NLP `class15/` · 42 scripts

Processamento de linguagem natural clássico e com deep learning.

| Código | Descrição |
|--------|-----------|
| GO1501–02 | Tokenização e normalização de texto |
| GO1503 | Pipeline completo de normalização |
| GO1504–05 | Stemming e lemmatização |
| GO1506–07 | Bag of Words (BoW) |
| GO1509 | N-gramas |
| GO1514–15 | Word2Vec |
| GO1516–17 | FastText e análise de subpalavras |
| GO1518 | Similaridade entre palavras |
| GO1520 | Embeddings em deep learning |
| GO1521 | Projeto: análise de sentimento |
| GO1522–24 | POS tagging e NER (completo com spaCy) |
| GO1525–26 | Classificação de texto com deep learning e CNN |
| GO1527–28 | Topic modeling com LDA |
| GO1529–31 | Sumarização, Question Answering |
| GO1532 | Tradução neural com MarianMT |
| GO1534–41 | Classificador de notícias, dataset IMDB, sistemas completos |

---

### Aula 16 — Transformers & LLMs `class16/` · 43 scripts

Arquitetura Transformer, BERT, GPT-2, T5, fine-tuning, LoRA, quantização e segurança.

| Código | Descrição |
|--------|-----------|
| GO1601 | Visualização e análise de attention |
| GO1602 | T5 — Multitask Learning |
| GO1603 | Chain-of-Thought Prompting |
| GO1604 | Fine-tuning end-to-end com BERT |
| GO1605 | Benchmarking e seleção de LLMs |
| GO1606 | LLMs multimodais — visão + linguagem |
| GO1607 | HuggingFace Transformers — guia prático |
| GO1609 | Comparação avançada de tokenizadores |
| GO1611 | Projeto: Chatbot com GPT-2 |
| GO1612 | Zero-shot e few-shot classification |
| GO1613 | Model distillation |
| GO1614 | Estratégias avançadas de sampling |
| GO1615 | LoRA — Low-Rank Adaptation |
| GO1616–17 | Quantização básica e técnicas avançadas |
| GO1619–20 | Embeddings e busca semântica |
| GO1621 | Estratégias para contextos longos |
| GO1622 | Calculadora de custo de LLMs |
| GO1623–25 | Detecção de alucinação, viés e defesa contra prompt injection |
| GO1627–36 | Exercícios avançados: self-attention, positional encoding, BERT vs GPT, fine-tuning, QA, geração de texto, sentence embeddings, MLM, LoRA, análise de attention weights |

---

### Aula 17 — LLMs Locais, Ollama & RAG `class17/` · 37 scripts

Execução local de LLMs, pipelines RAG, vector stores e aplicações conversacionais.

| Código | Descrição |
|--------|-----------|
| GO1701 | Vector stores |
| GO1704–05 | Metadata filtering, query routing e estratégias de chunking |
| GO1706 | Projeto: Chatbot RAG com Ollama |
| GO1707 | Monitoramento e observabilidade em produção |
| GO1708–10 | Query expansion, GraphRAG, otimização de custo |
| GO1712 | Agents com RAG |
| GO1713–15 | Streamlit UI, app RAG production-ready, QA e teste |
| GO1716 | Proteção contra prompt injection |
| GO1717 | Busca híbrida: BM25 + vector search |
| GO1718 | Métricas de avaliação de RAG |
| GO1719 | Streaming de respostas |
| GO1721 | Sistemas de memória conversacional |
| GO1723 | Comparação de vector stores: FAISS vs alternativas |

---

### Aula 18 — Reinforcement Learning `class18/` · 45 scripts

Q-Learning, DQN e algoritmos de RL moderno com Gymnasium.

| Código | Descrição |
|--------|-----------|
| GO1806 | Grid World com Q-Learning |
| GO1809 | DQN — setup |
| GO1815 | Ambientes Gymnasium |
| GO1817 | Projeto DQN no CartPole |
| GO1818 | Manipulação robótica, AlphaZero simplificado, data center, gestão de portfólio, recomendação |
| GO1819 | DQN no CartPole (exercício) |
| GO1820 | Comparação de algoritmos no LunarLander |
| GO1822–39 | Snippets: discretização, reward shaping, Q-tables, políticas |

---

### Aula 19 — Algoritmos Genéticos & Metaheurísticas `class19/` · 33 scripts

Algoritmos evolutivos, bio-inspirados e otimização combinatória.

| Código | Descrição |
|--------|-----------|
| GO1906 | Geração de população inicial |
| GO1907 | Projeto: maximizar função matemática com AG |
| GO1908 | Seleção por torneio e roleta |
| GO1909 | Função de Rosenbrock |
| GO1918 | Projeto TSP — Travelling Salesman Problem |
| GO1919 | Neuroevolução de redes neurais |
| GO1920 | Differential Evolution do zero |
| GO1921 | Particle Swarm Optimization (PSO) completo |
| GO1922 | Ant Colony Optimization (ACO) para TSP |
| GO1923 | CMA-ES com biblioteca |
| GO1924 | Genetic Programming — regressão simbólica |
| GO1925 | Benchmark comparativo de algoritmos |
| GO1926–28 | Exercícios: função multimodal, problema da mochila, tuning de hiperparâmetros |
| GO1929–30 | TSP com AG e Simulated Annealing |
| GO1931 | Otimização multiobjetivo |

---

### Aula 20 — MLOps & Deploy `class20/` · 20 scripts

Ciclo de vida de modelos em produção: rastreamento, API, monitoramento e CI/CD.

| Código | Descrição |
|--------|-----------|
| GO2001–04 | MLflow: tracking, UI, autolog, model registry |
| GO2005–07 | FastAPI: projeto Iris API, salvar/rodar modelo, testes |
| GO2008–09 | Azure Machine Learning — treinamento e inferência |
| GO2013 | Pipeline de retreino completo |
| GO2015 | Boas práticas de MLOps |
| GO2017 | Hyperparameter tuning avançado |
| GO2018 | Optuna — framework moderno de otimização |
| GO2019 | Learning rate scheduling |
| GO2020 | Model monitoring e drift detection |
| GO2021–22 | Testes automatizados com pytest |
| GO2023 | Flask + Prometheus para métricas |

---

### Aula 21 — IA Generativa & Futuro `class21/` · 25 scripts

GANs, VAEs, Stable Diffusion, XAI e tópicos avançados de ética e segurança em IA.

| Código | Descrição |
|--------|-----------|
| GO2020 | SHAP na prática; LIME para texto |
| GO2101 | GAN para MNIST |
| GO2102 | GAN completo com monitoramento |
| GO2103 | Stable Diffusion com Diffusers (HuggingFace) |
| GO2104 | VAE — Variational Autoencoder para MNIST |
| GO2105 | Stable Diffusion — geração avançada de imagens |
| GO2106 | VAE fine-tuning para dataset específico |
| GO2107–08 | XAI: GradCAM, SHAP e LIME |
| GO2109 | Fairness e mitigação de viés |
| GO2110 | Ataques adversariais e defesas |
| GO2111 | Differential Privacy |
| GO2112–16 | Exercícios avançados: GAN do zero, VAE, LIME, Stable Diffusion, pipeline MLOps completo |
| GO2121–23 | StarCoder, LLaMA, Mistral em execução local |

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

| Aula | Tema | Scripts |
|------|------|--------:|
| 01 | Introdução à IA & Ética | 22 |
| 02 | Representação do Conhecimento | 21 |
| 03 | Resolução de Problemas & Busca | 14 |
| 04 | Introdução ao Machine Learning | 32 |
| 05 | Algoritmos de Classificação | 43 |
| 06 | Regressão & Validação | 36 |
| 07 | Clustering | 14 |
| 08 | SOM — Mapas Auto-Organizáveis | 15 |
| 09 | Fundamentos de Redes Neurais | 30 |
| 10 | MLP — Redes Multicamadas | 53 |
| 11 | Lógica Fuzzy | 21 |
| 12 | CNNs — Redes Convolucionais | 60 |
| 13 | Visão Computacional & Reconhecimento de Padrões | 44 |
| 14 | RNN / LSTM | 41 |
| 15 | Introdução ao NLP | 42 |
| 16 | Transformers & LLMs | 43 |
| 17 | LLMs Locais, Ollama & RAG | 37 |
| 18 | Reinforcement Learning | 45 |
| 19 | Algoritmos Genéticos & Metaheurísticas | 33 |
| 20 | MLOps & Deploy | 20 |
| 21 | IA Generativa & Futuro | 25 |
| **Total** | | **691** |

---

## Licença

Material produzido para fins acadêmicos na disciplina de Inteligência Artificial — Pós-Graduação FAPA, 2026.
