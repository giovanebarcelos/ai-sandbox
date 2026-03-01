# GO1715-27bRagTestingQualityAssurance
import unittest
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass

@dataclass
class TestCase:
    """Caso de teste para RAG"""
    question: str
    expected_answer_keywords: List[str]
    expected_sources: List[str]
    min_relevance_score: float = 0.7
    max_latency_seconds: float = 5.0

class RAGTestSuite:
    """
    Suite de testes para sistemas RAG

    Testa:
    - Retrieval accuracy (documentos corretos recuperados?)
    - Answer quality (resposta contém keywords esperadas?)
    - Latency (tempo de resposta aceitável?)
    - Source attribution (fontes citadas corretamente?)
    - Edge cases (queries vazias, muito longas, etc.)
    """

    def __init__(self, rag_system):
        self.rag = rag_system
        self.test_results = []

    def test_retrieval_accuracy(self, test_cases: List[TestCase]) -> Dict:
        """Testa se documentos corretos são recuperados"""
        results = {
            'passed': 0,
            'failed': 0,
            'details': []
        }

        for tc in test_cases:
            # Query RAG
            retrieved = self.rag.retrieve(tc.question, k=5)
            retrieved_sources = [doc['metadata']['source'] for doc in retrieved]

            # Check if expected sources were retrieved
            found_sources = set(retrieved_sources) & set(tc.expected_sources)
            recall = len(found_sources) / len(tc.expected_sources) if tc.expected_sources else 1.0

            passed = recall >= 0.5  # At least 50% of expected sources found

            results['details'].append({
                'question': tc.question,
                'expected_sources': tc.expected_sources,
                'retrieved_sources': retrieved_sources,
                'recall': recall,
                'passed': passed
            })

            if passed:
                results['passed'] += 1
            else:
                results['failed'] += 1

        return results

    def test_answer_quality(self, test_cases: List[TestCase]) -> Dict:
        """Testa se respostas contém keywords esperadas"""
        results = {
            'passed': 0,
            'failed': 0,
            'details': []
        }

        for tc in test_cases:
            # Generate answer
            result = self.rag.query(tc.question)
            answer = result['answer'].lower()

            # Check keywords
            keywords_found = [kw for kw in tc.expected_answer_keywords 
                            if kw.lower() in answer]
            precision = len(keywords_found) / len(tc.expected_answer_keywords) \
                       if tc.expected_answer_keywords else 1.0

            passed = precision >= 0.6  # At least 60% keywords present

            results['details'].append({
                'question': tc.question,
                'answer': answer[:100] + '...',
                'expected_keywords': tc.expected_answer_keywords,
                'found_keywords': keywords_found,
                'precision': precision,
                'passed': passed
            })

            if passed:
                results['passed'] += 1
            else:
                results['failed'] += 1

        return results

    def test_latency(self, test_cases: List[TestCase]) -> Dict:
        """Testa se latência é aceitável"""
        import time

        results = {
            'passed': 0,
            'failed': 0,
            'latencies': [],
            'details': []
        }

        for tc in test_cases:
            start = time.time()
            result = self.rag.query(tc.question)
            latency = time.time() - start

            passed = latency <= tc.max_latency_seconds

            results['latencies'].append(latency)
            results['details'].append({
                'question': tc.question,
                'latency': latency,
                'max_allowed': tc.max_latency_seconds,
                'passed': passed
            })

            if passed:
                results['passed'] += 1
            else:
                results['failed'] += 1

        return results

    def test_edge_cases(self) -> Dict:
        """Testa casos extremos"""
        edge_cases = [
            "",  # Empty query
            "a",  # Very short query
            "what is " * 100,  # Very long query
            "askdjfhaksdjf",  # Gibberish
            "😀🎉🔥",  # Only emojis
            "SELECT * FROM users",  # SQL injection attempt
        ]

        results = {
            'passed': 0,
            'failed': 0,
            'details': []
        }

        for query in edge_cases:
            try:
                result = self.rag.query(query)
                # Should handle gracefully without crashing
                passed = 'answer' in result and result['answer'] is not None
            except Exception as e:
                passed = False
                error = str(e)

            results['details'].append({
                'query': query[:50] + '...' if len(query) > 50 else query,
                'passed': passed,
                'error': error if not passed else None
            })

            if passed:
                results['passed'] += 1
            else:
                results['failed'] += 1

        return results

    def run_full_test_suite(self, test_cases: List[TestCase]) -> Dict:
        """Executa todos os testes"""
        print("\n🧪 Executando Test Suite Completo\n")
        print("="*70)

        # Run all tests
        retrieval_results = self.test_retrieval_accuracy(test_cases)
        quality_results = self.test_answer_quality(test_cases)
        latency_results = self.test_latency(test_cases)
        edge_results = self.test_edge_cases()

        # Aggregate results
        total_tests = (
            len(test_cases) * 3 +  # retrieval, quality, latency
            len(edge_results['details'])
        )

        total_passed = (
            retrieval_results['passed'] +
            quality_results['passed'] +
            latency_results['passed'] +
            edge_results['passed']
        )

        print(f"\n📊 RESULTADOS FINAIS:")
        print(f"   Total testes: {total_tests}")
        print(f"   ✅ Passou: {total_passed} ({total_passed/total_tests*100:.1f}%)")
        print(f"   ❌ Falhou: {total_tests - total_passed}")

        print(f"\n📈 Detalhes por Categoria:")
        print(f"   Retrieval: {retrieval_results['passed']}/{len(test_cases)}")
        print(f"   Quality: {quality_results['passed']}/{len(test_cases)}")
        print(f"   Latency: {latency_results['passed']}/{len(test_cases)}")
        print(f"   Edge Cases: {edge_results['passed']}/{len(edge_results['details'])}")

        if latency_results['latencies']:
            print(f"\n⏱️  Latency Stats:")
            print(f"   Média: {np.mean(latency_results['latencies']):.2f}s")
            print(f"   P95: {np.percentile(latency_results['latencies'], 95):.2f}s")
            print(f"   P99: {np.percentile(latency_results['latencies'], 99):.2f}s")

        return {
            'retrieval': retrieval_results,
            'quality': quality_results,
            'latency': latency_results,
            'edge_cases': edge_results,
            'summary': {
                'total_tests': total_tests,
                'passed': total_passed,
                'failed': total_tests - total_passed,
                'pass_rate': total_passed / total_tests
            }
        }

# === EXEMPLO DE USO ===

# Mock RAG para testes
class MockRAGForTesting:
    def retrieve(self, query, k=5):
        return [
            {'text': 'Machine learning...', 'metadata': {'source': 'ml_intro.pdf'}},
            {'text': 'Neural networks...', 'metadata': {'source': 'neural_nets.pdf'}}
        ]

    def query(self, question):
        import time
        time.sleep(0.2)  # Simulate latency
        return {
            'answer': 'Machine learning is a subset of AI that enables computers to learn from data.',
            'sources': ['ml_intro.pdf']
        }

# Create test cases


if __name__ == "__main__":
    test_cases = [
        TestCase(
            question="What is machine learning?",
            expected_answer_keywords=["machine", "learning", "data", "AI"],
            expected_sources=["ml_intro.pdf"],
            max_latency_seconds=2.0
        ),
        TestCase(
            question="How do neural networks work?",
            expected_answer_keywords=["neural", "network", "layers", "neurons"],
            expected_sources=["neural_nets.pdf"],
            max_latency_seconds=2.0
        ),
        TestCase(
            question="Explain transformers in NLP",
            expected_answer_keywords=["transformer", "attention", "NLP"],
            expected_sources=["transformers.pdf"],
            max_latency_seconds=2.0
        )
    ]

    # Run tests
    rag = MockRAGForTesting()
    test_suite = RAGTestSuite(rag)
    results = test_suite.run_full_test_suite(test_cases)

    # Visualize results
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Pass/Fail by category
    ax = axes[0, 0]
    categories = ['Retrieval', 'Quality', 'Latency', 'Edge Cases']
    passed = [
        results['retrieval']['passed'],
        results['quality']['passed'],
        results['latency']['passed'],
        results['edge_cases']['passed']
    ]
    failed = [
        results['retrieval']['failed'],
        results['quality']['failed'],
        results['latency']['failed'],
        results['edge_cases']['failed']
    ]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(x - width/2, passed, width, label='Passed', color='lightgreen', alpha=0.8)
    ax.bar(x + width/2, failed, width, label='Failed', color='lightcoral', alpha=0.8)
    ax.set_ylabel('Count')
    ax.set_title('Test Results by Category')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 2. Overall pass rate
    ax = axes[0, 1]
    pass_rate = results['summary']['pass_rate']
    ax.pie([pass_rate, 1-pass_rate], labels=['Passed', 'Failed'],
           colors=['lightgreen', 'lightcoral'], autopct='%1.1f%%', startangle=90)
    ax.set_title(f"Overall Pass Rate\n({results['summary']['passed']}/{results['summary']['total_tests']})")

    # 3. Latency distribution
    ax = axes[1, 0]
    latencies = results['latency']['latencies']
    ax.hist(latencies, bins=10, color='skyblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(latencies), color='red', linestyle='--', label=f'Mean: {np.mean(latencies):.2f}s')
    ax.set_xlabel('Latency (seconds)')
    ax.set_ylabel('Frequency')
    ax.set_title('Latency Distribution')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 4. Test coverage
    ax = axes[1, 1]
    test_types = ['Functional', 'Performance', 'Edge Cases']
    test_counts = [
        len(test_cases) * 2,  # retrieval + quality
        len(test_cases),  # latency
        len(results['edge_cases']['details'])
    ]

    ax.barh(test_types, test_counts, color='purple', alpha=0.7)
    ax.set_xlabel('Number of Tests')
    ax.set_title('Test Coverage')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('rag_test_results.png', dpi=150, bbox_inches='tight')
    print("\n📊 Gráfico salvo: rag_test_results.png")

    print("\n✅ RAG Testing Suite implementado!")
