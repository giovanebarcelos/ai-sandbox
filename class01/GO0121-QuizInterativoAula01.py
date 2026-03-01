# GO0121-QuizInterativoAula01
import random
import time

# ═══════════════════════════════════════════════════════════════════
# QUIZ INTERATIVO - AULA 01
# ═══════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    print("="*70)
    print("QUIZ INTERATIVO: INTRODUÇÃO À IA E ÉTICA")
    print("Teste seu conhecimento sobre a Aula 01!")
    print("="*70)

    class QuizIA:
        """Quiz interativo sobre conceitos da Aula 01"""

        def __init__(self):
            self.questoes = self._criar_questoes()
            self.pontuacao = 0
            self.total_questoes = len(self.questoes)
            self.respostas_usuario = []

        def _criar_questoes(self):
            """Base de questões sobre Aula 01"""
            return [
                {
                    "id": 1,
                    "categoria": "História",
                    "pergunta": "Qual foi o marco que deu origem oficial à IA como campo?",
                    "opcoes": [
                        "A) Publicação do artigo de Turing (1950)",
                        "B) Conferência de Dartmouth (1956)",
                        "C) Invenção do Perceptron (1958)",
                        "D) Vitória do Deep Blue (1997)"
                    ],
                    "resposta_correta": "B",
                    "explicacao": "A Conferência de Dartmouth em 1956 é considerada o nascimento oficial da IA como campo de pesquisa, cunhando o termo 'Inteligência Artificial'.",
                    "conceito": "O campo de IA tem marcos históricos importantes que definiram sua evolução."
                },
                {
                    "id": 2,
                    "categoria": "Ética",
                    "pergunta": "No dilema do carro autônomo, qual framework ético escolheria 'EVITAR ação ativa de matar 1 pessoa' mesmo que isso resulte em 5 mortes passivas?",
                    "opcoes": [
                        "A) Utilitarismo (maximizar bem-estar)",
                        "B) Deontologia (seguir regras absolutas)",
                        "C) Ética de Virtude (agir virtuosamente)",
                        "D) Consequencialismo (avaliar resultados)"
                    ],
                    "resposta_correta": "B",
                    "explicacao": "Deontologia (Kant) segue regras absolutas como 'não matar ativamente', mesmo que o resultado seja pior no total.",
                    "conceito": "Diferentes frameworks éticos produzem decisões diferentes para o mesmo dilema."
                },
                {
                    "id": 3,
                    "categoria": "Viés",
                    "pergunta": "Qual métrica indica viés significativo se estiver ABAIXO de 0.8 (80%)?",
                    "opcoes": [
                        "A) Acurácia",
                        "B) Precisão",
                        "C) Disparate Impact Ratio",
                        "D) F1-Score"
                    ],
                    "resposta_correta": "C",
                    "explicacao": "Disparate Impact Ratio < 0.8 indica viés significativo (regra dos 80%), comparando taxas de aprovação entre grupos.",
                    "conceito": "Fairness metrics quantificam discriminação em modelos de ML."
                },
                {
                    "id": 4,
                    "categoria": "AGI",
                    "pergunta": "Qual desafio o GPT-4 FALHA drasticamente (humanos 85%, GPT-4 <5%)?",
                    "opcoes": [
                        "A) Geração de texto fluente",
                        "B) Raciocínio abstrato (ARC-AGI)",
                        "C) Tradução entre idiomas",
                        "D) Summarização de documentos"
                    ],
                    "resposta_correta": "B",
                    "explicacao": "ARC-AGI testa raciocínio abstrato com quebra-cabeças visuais. Humanos: 85%, GPT-4: <5%. Gap enorme!",
                    "conceito": "LLMs dominam linguagem mas falham em raciocínio abstrato profundo."
                },
                {
                    "id": 5,
                    "categoria": "XAI",
                    "pergunta": "Qual técnica de XAI explica decisões LOCAIS (instância específica)?",
                    "opcoes": [
                        "A) Feature Importance (importância global)",
                        "B) Partial Dependence Plot (efeito global)",
                        "C) LIME (explicação local)",
                        "D) Confusion Matrix (avaliação geral)"
                    ],
                    "resposta_correta": "C",
                    "explicacao": "LIME (Local Interpretable Model-Agnostic Explanations) explica POR QUÊ uma instância específica foi classificada de determinada forma.",
                    "conceito": "XAI tem técnicas globais (modelo todo) e locais (instância individual)."
                },
                {
                    "id": 6,
                    "categoria": "Benchmarks",
                    "pergunta": "Quanto MMLU (conhecimento multidisciplinar) evoluiu de 2021 a 2024?",
                    "opcoes": [
                        "A) 10 pontos (43.9% → 53%)",
                        "B) 20 pontos (43.9% → 63%)",
                        "C) 46 pontos (43.9% → 90%)",
                        "D) 5 pontos (43.9% → 48%)"
                    ],
                    "resposta_correta": "C",
                    "explicacao": "MMLU teve salto GIGANTE: GPT-3 (2021) 43.9% → Gemini Ultra (2024) 90%. Progresso de 46 pontos em 3 anos!",
                    "conceito": "Benchmarks mostram progresso acelerado em algumas áreas (especialmente conhecimento)."
                },
                {
                    "id": 7,
                    "categoria": "Turing Test",
                    "pergunta": "Qual é a crítica filosófica CENTRAL ao Teste de Turing?",
                    "opcoes": [
                        "A) É muito fácil de passar",
                        "B) Imitar ≠ Inteligir (Chinese Room - Searle)",
                        "C) Não mede velocidade de processamento",
                        "D) Só funciona em inglês"
                    ],
                    "resposta_correta": "B",
                    "explicacao": "Chinese Room (Searle): Manipular símbolos (sintaxe) não é o mesmo que entender significado (semântica). Imitar comportamento ≠ ter consciência.",
                    "conceito": "Debate filosófico: Comportamento inteligente implica consciência/entendimento?"
                },
                {
                    "id": 8,
                    "categoria": "Futuro do Trabalho",
                    "pergunta": "Qual característica de trabalho tem MENOR risco de automação?",
                    "opcoes": [
                        "A) Tarefas repetitivas e padronizadas",
                        "B) Seguir scripts e procedimentos",
                        "C) Criatividade e empatia profunda",
                        "D) Processamento de dados estruturados"
                    ],
                    "resposta_correta": "C",
                    "explicacao": "Criatividade original e empatia profunda são difíceis de automatizar. Psicólogo (15%), Professor (25%), Artista (30%) têm baixo risco.",
                    "conceito": "Habilidades humanas únicas (criatividade, empatia, raciocínio complexo) são mais protegidas."
                },
                {
                    "id": 9,
                    "categoria": "Paradigmas",
                    "pergunta": "Qual é a principal VANTAGEM da IA Simbólica sobre Conexionista?",
                    "opcoes": [
                        "A) Aprende automaticamente de dados",
                        "B) Lida bem com ruído e variações",
                        "C) Interpretabilidade (regras legíveis)",
                        "D) Escalabilidade para big data"
                    ],
                    "resposta_correta": "C",
                    "explicacao": "IA Simbólica (regras IF-THEN) é transparente e interpretável. 'SE febre E tosse ENTÃO gripe' é legível. Neural networks são black-box.",
                    "conceito": "Trade-off: Simbólica (interpretável mas frágil) vs Conexionista (precisa mas opaca)."
                },
                {
                    "id": 10,
                    "categoria": "Regulação",
                    "pergunta": "Qual legislação garante 'direito à explicação' de decisões automatizadas?",
                    "opcoes": [
                        "A) HIPAA (EUA - saúde)",
                        "B) GDPR Art. 22 (EU) e LGPD Art. 20 (BR)",
                        "C) Fair Credit Reporting Act (EUA - crédito)",
                        "D) Não existe ainda"
                    ],
                    "resposta_correta": "B",
                    "explicacao": "GDPR (EU) Art. 22 e LGPD (Brasil) Art. 20 garantem direito de contestar e obter explicação de decisões automatizadas que impactem significativamente.",
                    "conceito": "Regulação está emergindo para exigir transparência em sistemas de IA críticos."
                }
            ]

        def exibir_questao(self, questao):
            """Exibe uma questão formatada"""
            print(f"\n{'='*70}")
            print(f"QUESTÃO {questao['id']}/{self.total_questoes} - Categoria: {questao['categoria']}")
            print(f"{'='*70}")
            print(f"\n{questao['pergunta']}\n")

            for opcao in questao['opcoes']:
                print(f"   {opcao}")

            print()

        def coletar_resposta(self):
            """Coleta resposta do usuário"""
            while True:
                resposta = input("Sua resposta (A/B/C/D): ").strip().upper()
                if resposta in ['A', 'B', 'C', 'D']:
                    return resposta
                print("⚠️ Resposta inválida. Use A, B, C ou D.")

        def verificar_resposta(self, questao, resposta_usuario):
            """Verifica se resposta está correta"""
            correta = questao['resposta_correta']
            acertou = (resposta_usuario == correta)

            if acertou:
                print("✅ CORRETO!")
                self.pontuacao += 1
            else:
                print(f"❌ INCORRETO. Resposta correta: {correta}")

            print(f"\n💡 Explicação:")
            print(f"   {questao['explicacao']}")
            print(f"\n📚 Conceito-chave:")
            print(f"   {questao['conceito']}")

            # Registrar
            self.respostas_usuario.append({
                "questao_id": questao['id'],
                "categoria": questao['categoria'],
                "resposta_usuario": resposta_usuario,
                "resposta_correta": correta,
                "acertou": acertou
            })

            time.sleep(2)  # Pausa para leitura

        def executar_quiz(self):
            """Executa o quiz completo"""
            print("\n🎮 Começando o quiz em 3 segundos...\n")
            time.sleep(3)

            # Randomizar ordem das questões
            questoes_aleatorias = self.questoes.copy()
            random.shuffle(questoes_aleatorias)

            for questao in questoes_aleatorias:
                self.exibir_questao(questao)
                resposta = self.coletar_resposta()
                self.verificar_resposta(questao, resposta)

            self.exibir_resultado_final()

        def exibir_resultado_final(self):
            """Exibe resultado final do quiz"""
            print("\n" + "="*70)
            print("RESULTADO FINAL")
            print("="*70)

            percentual = (self.pontuacao / self.total_questoes) * 100

            print(f"\n📊 Pontuação: {self.pontuacao}/{self.total_questoes} ({percentual:.0f}%)")

            # Feedback por faixa
            if percentual >= 90:
                emoji = "🏆"
                feedback = "EXCELENTE! Você domina os conceitos da Aula 01!"
            elif percentual >= 70:
                emoji = "🎉"
                feedback = "MUITO BOM! Boa compreensão dos conceitos."
            elif percentual >= 50:
                emoji = "👍"
                feedback = "BOM! Continue revisando os conceitos."
            else:
                emoji = "📖"
                feedback = "PRECISA REVISAR. Estude os slides novamente."

            print(f"\n{emoji} {feedback}")

            # Análise por categoria
            print(f"\n📈 DESEMPENHO POR CATEGORIA:")
            categorias_stats = {}

            for resposta in self.respostas_usuario:
                cat = resposta['categoria']
                if cat not in categorias_stats:
                    categorias_stats[cat] = {'acertos': 0, 'total': 0}
                categorias_stats[cat]['total'] += 1
                if resposta['acertou']:
                    categorias_stats[cat]['acertos'] += 1

            for categoria, stats in sorted(categorias_stats.items()):
                acertos = stats['acertos']
                total = stats['total']
                perc_cat = (acertos / total) * 100

                if perc_cat == 100:
                    emoji_cat = "✅"
                elif perc_cat >= 50:
                    emoji_cat = "🟡"
                else:
                    emoji_cat = "❌"

                print(f"   {emoji_cat} {categoria}: {acertos}/{total} ({perc_cat:.0f}%)")

            # Recomendações
            print(f"\n💡 RECOMENDAÇÕES:")

            categorias_fracas = [cat for cat, stats in categorias_stats.items() 
                                if (stats['acertos'] / stats['total']) < 0.5]

            if categorias_fracas:
                print(f"   → Revisar especialmente: {', '.join(categorias_fracas)}")
            else:
                print(f"   → Continue praticando e aprofundando!")

            if percentual < 70:
                print(f"   → Rever slides da Aula 01")
                print(f"   → Executar os códigos práticos novamente")

            print(f"\n✅ Quiz concluído! Obrigado por participar.")

    # ═══════════════════════════════════════════════════════════════════
    # EXECUTAR QUIZ
    # ═══════════════════════════════════════════════════════════════════

    quiz = QuizIA()
    quiz.executar_quiz()

    print("\n" + "="*70)
    print("🎓 AULA 01 - INTRODUÇÃO À IA E ÉTICA - FINALIZADA")
    print("="*70)
    print("\n📚 Conceitos aprendidos:")
    print("   ✓ História da IA (Dartmouth, Invernos, LLMs)")
    print("   ✓ Ética e Viés (fairness metrics, dilemas morais)")
    print("   ✓ Benchmarks (ImageNet, MMLU, HumanEval)")
    print("   ✓ AGI (critérios, status atual, obstáculos)")
    print("   ✓ XAI (LIME, SHAP, interpretabilidade)")
    print("   ✓ Futuro do Trabalho (automação, habilidades)")
    print("   ✓ Paradigmas (Simbólica vs Conexionista)")
    print("   ✓ Regulação (GDPR, LGPD)")

    print("\n🚀 Próximos passos:")
    print("   → Aula 02: Representação de Conhecimento")
    print("   → Continue praticando os exercícios")
    print("   → Explore ferramentas (SHAP, LIME, RAG)")

    print("\n" + "="*70)
