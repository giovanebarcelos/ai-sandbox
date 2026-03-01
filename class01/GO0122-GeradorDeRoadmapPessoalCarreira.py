# GO0122-GeradorDeRoadmapPessoalCarreira
import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# GERADOR DE ROADMAP PESSOAL - CARREIRA EM IA
# ═══════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    print("="*70)
    print("ROADMAP PESSOAL: SUA JORNADA EM IA")
    print("="*70)

    class GeradorRoadmap:
        """Gera roadmap personalizado de aprendizado em IA"""

        def __init__(self):
            self.perfis = self._definir_perfis()
            self.recursos = self._definir_recursos()

        def _definir_perfis(self):
            """Define perfis de carreira em IA"""
            return {
                "1": {
                    "nome": "ML Engineer",
                    "descricao": "Implementa modelos de ML em produção",
                    "fundamentos": ["Python", "Estatística", "Álgebra Linear"],
                    "principais": [
                        "Scikit-learn, TensorFlow, PyTorch",
                        "MLOps (Docker, Kubernetes, MLflow)",
                        "Cloud (AWS SageMaker, GCP AI Platform)",
                        "Feature Engineering e Data Pipelines"
                    ],
                    "avancados": [
                        "Model Optimization (quantização, pruning)",
                        "Distributed Training",
                        "Edge AI (TensorFlow Lite)",
                        "AutoML"
                    ],
                    "projetos": [
                        "Sistema de recomendação (Netflix-like)",
                        "Detector de fraude (credit card)",
                        "Chatbot com RAG",
                        "Deploy modelo em produção (API + monitoring)"
                    ],
                    "salario_br": "R$ 8.000 - R$ 20.000",
                    "demanda": "🔥🔥🔥 Altíssima"
                },
                "2": {
                    "nome": "Data Scientist",
                    "descricao": "Analisa dados e treina modelos para insights",
                    "fundamentos": ["Python/R", "Estatística", "Matemática"],
                    "principais": [
                        "Pandas, NumPy, Matplotlib",
                        "Scikit-learn, XGBoost",
                        "A/B Testing e Experimentação",
                        "SQL e Bancos de Dados"
                    ],
                    "avancados": [
                        "Causal Inference",
                        "Bayesian Methods",
                        "Time Series Forecasting",
                        "Deep Learning (opcional)"
                    ],
                    "projetos": [
                        "Análise exploratória de dados (EDA)",
                        "Churn prediction (telecom)",
                        "Precificação dinâmica (Uber-like)",
                        "Dashboard interativo (Streamlit/Dash)"
                    ],
                    "salario_br": "R$ 7.000 - R$ 18.000",
                    "demanda": "🔥🔥 Alta"
                },
                "3": {
                    "nome": "AI Researcher",
                    "descricao": "Desenvolve novos algoritmos e publica papers",
                    "fundamentos": ["Matemática Avançada", "Programação", "Inglês"],
                    "principais": [
                        "Deep Learning (Transformers, GANs, Diffusion)",
                        "Reinforcement Learning",
                        "Paper Reading (ArXiv)",
                        "Implementação de Papers"
                    ],
                    "avancados": [
                        "Novel Architectures",
                        "Theoretical ML",
                        "Optimization Algorithms",
                        "AGI Research"
                    ],
                    "projetos": [
                        "Reimplementar paper famoso (Attention Is All You Need)",
                        "Propor melhoria em arquitetura existente",
                        "Participar de competições (Kaggle, NeurIPS)",
                        "Publicar em conferência (ICML, NeurIPS, ICLR)"
                    ],
                    "salario_br": "R$ 12.000 - R$ 30.000+",
                    "demanda": "🔥 Moderada (vagas seniores)"
                },
                "4": {
                    "nome": "NLP Engineer",
                    "descricao": "Especialista em processamento de linguagem natural",
                    "fundamentos": ["Linguística", "Python", "Transformers"],
                    "principais": [
                        "Hugging Face Transformers",
                        "Fine-tuning LLMs (LoRA, QLoRA)",
                        "Prompt Engineering",
                        "RAG (Retrieval Augmented Generation)"
                    ],
                    "avancados": [
                        "LLM Alignment (RLHF)",
                        "Multi-modal Models (CLIP, GPT-4V)",
                        "Efficient Fine-tuning",
                        "LLM Serving (vLLM, TGI)"
                    ],
                    "projetos": [
                        "Chatbot com contexto (RAG + LangChain)",
                        "Fine-tune modelo para domínio específico",
                        "Sistema de Q&A sobre documentos",
                        "Sentiment analysis em tempo real"
                    ],
                    "salario_br": "R$ 9.000 - R$ 22.000",
                    "demanda": "🔥🔥🔥 Altíssima (boom LLMs)"
                },
                "5": {
                    "nome": "Computer Vision Engineer",
                    "descricao": "Especialista em visão computacional",
                    "fundamentos": ["Processamento de Imagens", "CNNs", "Python"],
                    "principais": [
                        "OpenCV, PIL",
                        "PyTorch/TensorFlow para Vision",
                        "Object Detection (YOLO, Faster R-CNN)",
                        "Segmentation (U-Net, Mask R-CNN)"
                    ],
                    "avancados": [
                        "Vision Transformers (ViT)",
                        "3D Vision (NeRF, Gaussian Splatting)",
                        "Video Understanding",
                        "Edge Deployment"
                    ],
                    "projetos": [
                        "Detector de objetos em tempo real",
                        "Segmentação semântica (carros autônomos)",
                        "Face recognition system",
                        "Anomaly detection em manufatura"
                    ],
                    "salario_br": "R$ 9.000 - R$ 22.000",
                    "demanda": "🔥🔥 Alta (indústria, segurança)"
                },
                "6": {
                    "nome": "AI Ethics & Governance",
                    "descricao": "Garante IA responsável e ética",
                    "fundamentos": ["Filosofia", "Ética", "Regulação"],
                    "principais": [
                        "Fairness Metrics (Disparate Impact, Equalized Odds)",
                        "XAI (LIME, SHAP)",
                        "Bias Auditing",
                        "GDPR, LGPD, AI Act (EU)"
                    ],
                    "avancados": [
                        "AI Safety & Alignment",
                        "Red Teaming LLMs",
                        "Constitutional AI",
                        "Policy Development"
                    ],
                    "projetos": [
                        "Auditoria de viés em sistema real",
                        "Framework de governança para empresa",
                        "Relatório de impacto algorítmico",
                        "Advocacy e educação pública"
                    ],
                    "salario_br": "R$ 8.000 - R$ 20.000",
                    "demanda": "🔥 Crescente (regulação)"
                }
            }

        def _definir_recursos(self):
            """Define recursos de aprendizado"""
            return {
                "cursos_online": {
                    "Básico": [
                        "CS50's Introduction to AI (Harvard) - GRATUITO",
                        "Machine Learning Specialization (Coursera - Andrew Ng)",
                        "Fast.ai - Practical Deep Learning"
                    ],
                    "Intermediário": [
                        "Deep Learning Specialization (Coursera)",
                        "Fullstack Deep Learning (Berkeley)",
                        "MLOps Specialization (DeepLearning.AI)"
                    ],
                    "Avançado": [
                        "Stanford CS231n (Computer Vision)",
                        "Stanford CS224n (NLP)",
                        "Berkeley CS285 (Deep RL)"
                    ]
                },
                "livros": {
                    "Fundamentos": [
                        "Hands-On Machine Learning (Aurélien Géron)",
                        "Deep Learning (Goodfellow, Bengio, Courville)",
                        "Pattern Recognition and Machine Learning (Bishop)"
                    ],
                    "Específicos": [
                        "Speech and Language Processing (Jurafsky & Martin) - NLP",
                        "Reinforcement Learning (Sutton & Barto)",
                        "Computer Vision: Algorithms and Applications (Szeliski)"
                    ]
                },
                "pratica": [
                    "Kaggle (competições)",
                    "GitHub (contribuir para projetos open source)",
                    "Papers with Code (implementar papers)",
                    "Hugging Face (modelos e datasets)"
                ],
                "comunidades": [
                    "r/MachineLearning (Reddit)",
                    "Hugging Face Discord",
                    "AI Brasil (Telegram)",
                    "Meetups locais"
                ]
            }

        def escolher_perfil(self):
            """Interação para escolher perfil"""
            print("\n📋 ESCOLHA SEU PERFIL DE CARREIRA EM IA:\n")

            for codigo, perfil in self.perfis.items():
                print(f"{codigo}. {perfil['nome']}")
                print(f"   {perfil['descricao']}")
                print(f"   Salário BR: {perfil['salario_br']}")
                print(f"   Demanda: {perfil['demanda']}\n")

            while True:
                escolha = input("Digite o número do perfil desejado (1-6): ").strip()
                if escolha in self.perfis:
                    return escolha
                print("⚠️ Escolha inválida. Use 1, 2, 3, 4, 5 ou 6.")

        def gerar_roadmap(self, perfil_codigo):
            """Gera roadmap detalhado"""
            perfil = self.perfis[perfil_codigo]

            print("\n" + "="*70)
            print(f"ROADMAP: {perfil['nome'].upper()}")
            print("="*70)
            print(f"\n{perfil['descricao']}")
            print(f"\n💰 Salário no Brasil: {perfil['salario_br']}")
            print(f"📈 Demanda de Mercado: {perfil['demanda']}")

            # Fase 1: Fundamentos (3-6 meses)
            print("\n" + "="*70)
            print("FASE 1: FUNDAMENTOS (3-6 meses)")
            print("="*70)
            print("\n🎯 Objetivos:")
            for fund in perfil['fundamentos']:
                print(f"   ✓ {fund}")

            print("\n📚 Recursos:")
            for curso in self.recursos['cursos_online']['Básico']:
                print(f"   • {curso}")
            print(f"   • Livro: {self.recursos['livros']['Fundamentos'][0]}")

            # Fase 2: Principais Habilidades (6-12 meses)
            print("\n" + "="*70)
            print("FASE 2: HABILIDADES PRINCIPAIS (6-12 meses)")
            print("="*70)
            print("\n🎯 Objetivos:")
            for hab in perfil['principais']:
                print(f"   ✓ {hab}")

            print("\n📚 Recursos:")
            for curso in self.recursos['cursos_online']['Intermediário']:
                print(f"   • {curso}")

            print("\n💼 Projetos desta fase:")
            for i, projeto in enumerate(perfil['projetos'][:2], 1):
                print(f"   {i}. {projeto}")

            # Fase 3: Avançado (12-24 meses)
            print("\n" + "="*70)
            print("FASE 3: HABILIDADES AVANÇADAS (12-24 meses)")
            print("="*70)
            print("\n🎯 Objetivos:")
            for av in perfil['avancados']:
                print(f"   ✓ {av}")

            print("\n📚 Recursos:")
            for curso in self.recursos['cursos_online']['Avançado']:
                print(f"   • {curso}")

            print("\n💼 Projetos desta fase:")
            for i, projeto in enumerate(perfil['projetos'][2:], 3):
                print(f"   {i}. {projeto}")

            # Prática contínua
            print("\n" + "="*70)
            print("PRÁTICA CONTÍNUA (sempre)")
            print("="*70)
            for pratica in self.recursos['pratica']:
                print(f"   • {pratica}")

            # Comunidades
            print("\n" + "="*70)
            print("COMUNIDADES (networking)")
            print("="*70)
            for com in self.recursos['comunidades']:
                print(f"   • {com}")

            # Visualização do roadmap
            self._visualizar_roadmap(perfil)

            # Próximos passos
            print("\n" + "="*70)
            print("PRÓXIMOS PASSOS (COMEÇAR HOJE!)")
            print("="*70)
            print("1️⃣ Escolher 1 curso da Fase 1 e começar HOJE")
            print("2️⃣ Configurar ambiente Python (Anaconda ou venv)")
            print("3️⃣ Criar conta no GitHub e Kaggle")
            print("4️⃣ Definir horário diário de estudo (1-2h mínimo)")
            print("5️⃣ Fazer primeiro projeto simples (Iris classification, MNIST)")
            print("6️⃣ Entrar em 1 comunidade (Reddit ou Telegram)")

            print("\n💡 DICAS DE OURO:")
            print("   • Consistência > Intensidade (1h todo dia > 7h 1 vez/semana)")
            print("   • Projeto > Certificado (portfolio no GitHub vale mais)")
            print("   • Ensinar = Aprender (escreva blog posts, faça vídeos)")
            print("   • Network (comunidades, meetups, LinkedIn)")
            print("   • Paciência (leva 2-3 anos para ficar proficiente)")

            print("\n✅ Roadmap personalizado gerado!")

        def _visualizar_roadmap(self, perfil):
            """Visualiza roadmap como gráfico"""
            print("\n📊 GERANDO VISUALIZAÇÃO DO ROADMAP...")

            fig, axes = plt.subplots(2, 1, figsize=(14, 10))

            # 1. Timeline
            ax1 = axes[0]
            fases = ['Fundamentos\n(3-6 meses)', 'Principais\n(6-12 meses)', 
                    'Avançado\n(12-24 meses)', 'Especialista\n(24+ meses)']
            meses = [4.5, 9, 18, 30]  # Ponto médio de cada fase
            competencia = [30, 60, 85, 95]  # % competência

            ax1.plot(meses, competencia, 'o-', linewidth=3, markersize=12, 
                    color='green', label='Progresso Esperado')
            ax1.fill_between(meses, competencia, alpha=0.3, color='green')

            # Anotar fases
            for i, (mes, comp, fase) in enumerate(zip(meses, competencia, fases)):
                ax1.annotate(fase, xy=(mes, comp), xytext=(mes, comp + 10),
                            ha='center', fontsize=9,
                            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

            ax1.set_xlabel("Meses de Estudo", fontsize=12)
            ax1.set_ylabel("Competência (%)", fontsize=12)
            ax1.set_title(f"Timeline de Aprendizado: {perfil['nome']}", 
                         fontsize=13, fontweight='bold')
            ax1.set_ylim(0, 110)
            ax1.set_xlim(0, 36)
            ax1.axhline(y=70, color='orange', linestyle='--', label='Pronto para mercado (70%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. Habilidades por fase
            ax2 = axes[1]
            habilidades_labels = ['Fundamentos', 'Principais', 'Avançados', 'Soft Skills']
            num_hab = [len(perfil['fundamentos']), 
                      len(perfil['principais']), 
                      len(perfil['avancados']),
                      4]  # Soft skills: comunicação, trabalho em equipe, resolução de problemas, aprendizado contínuo

            colors = ['lightblue', 'skyblue', 'royalblue', 'gold']
            bars = ax2.barh(habilidades_labels, num_hab, color=colors, alpha=0.7)

            # Adicionar valores nas barras
            for i, (bar, val) in enumerate(zip(bars, num_hab)):
                ax2.text(val + 0.2, bar.get_y() + bar.get_height()/2, 
                        str(val), va='center', fontsize=11, fontweight='bold')

            ax2.set_xlabel("Número de Habilidades", fontsize=12)
            ax2.set_title("Habilidades a Dominar", fontsize=13, fontweight='bold')
            ax2.set_xlim(0, max(num_hab) + 2)
            ax2.grid(True, alpha=0.3, axis='x')

            plt.suptitle(f"Roadmap Personalizado: {perfil['nome']}", 
                        fontsize=15, fontweight='bold')
            plt.tight_layout()
            plt.show()

    # ═══════════════════════════════════════════════════════════════════
    # EXECUTAR GERADOR DE ROADMAP
    # ═══════════════════════════════════════════════════════════════════

    gerador = GeradorRoadmap()
    perfil_escolhido = gerador.escolher_perfil()
    gerador.gerar_roadmap(perfil_escolhido)

    print("\n" + "="*70)
    print("🎓 BOA JORNADA EM IA!")
    print("="*70)
    print("\n💬 Lembre-se:")
    print("   'A jornada de mil milhas começa com um único passo.' - Lao Tzu")
    print("   'Não é sobre ser o melhor, é sobre ser melhor que ontem.' - Anônimo")
    print("\n🚀 Comece HOJE. Seu futuro você agradece.")
    print("="*70)
