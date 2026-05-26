# GO1704-16aAdvancedMetadataFilteringQueryRouting
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict
import re

import matplotlib
import matplotlib.pyplot as plt

# Garante exibição inline em Colab/Jupyter mesmo que o backend tenha sido
# alterado em sessões anteriores (ex: Agg definido e kernel não reiniciado)
import matplotlib
matplotlib.use('Agg')  # Backend sem interface gráfica (compatível com servidor/script)

class SmartQueryRouter:
    """
    Router inteligente que:
    1. Analisa queries para extrair intenção
    2. Determina filtros de metadata automaticamente
    3. Roteia para diferentes índices/coleções
    4. Aplica regras de negócio específicas

    Casos de uso:
    - Multi-tenant systems
    - Time-based filtering
    - Permission-based access
    - Document type routing
    """

    def __init__(self):
        self.routes = {}
        self.metadata_extractors = []
        self.routing_stats = defaultdict(int)

    def register_route(self, name: str, condition_fn, index_name: str):
        """Registra rota com condição e índice de destino"""
        self.routes[name] = {
            'condition': condition_fn,
            'index': index_name,
            'hits': 0
        }

    def add_metadata_extractor(self, extractor_fn):
        """Adiciona função para extrair metadata de query"""
        self.metadata_extractors.append(extractor_fn)

    def extract_date_filter(self, query: str) -> Dict:
        """Extrai filtros temporais da query"""
        filters = {}

        # Padrões temporais
        patterns = {
            'today': lambda: datetime.now().date(),
            'ontem|yesterday': lambda: (datetime.now() - timedelta(days=1)).date(),
            r'(\d{4})': lambda m: {'year': int(m.group(1))},
            r'últimos (\d+) dias': lambda m: {
                'date_from': (datetime.now() - timedelta(days=int(m.group(1)))).isoformat()
            },
            r'(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)': 
                lambda m: {'month': self._month_to_number(m.group(1))}
        }

        query_lower = query.lower()
        for pattern, extractor in patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                result = extractor() if callable(extractor) and not match.groups() else extractor(match)
                if isinstance(result, dict):
                    filters.update(result)
                else:
                    filters['date'] = result.isoformat()

        return filters

    def _month_to_number(self, month_name: str) -> int:
        months = {
            'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4,
            'maio': 5, 'junho': 6, 'julho': 7, 'agosto': 8,
            'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12
        }
        return months.get(month_name.lower(), 0)

    def extract_document_type(self, query: str) -> Dict:
        """Identifica tipo de documento mencionado"""
        doc_types = {
            'manual|guia|tutorial': 'documentation',
            'relatório|report': 'report',
            'política|policy': 'policy',
            'código|code|script': 'code',
            'email|mensagem': 'email',
            'contrato|agreement': 'contract'
        }

        query_lower = query.lower()
        for pattern, doc_type in doc_types.items():
            if re.search(pattern, query_lower):
                return {'document_type': doc_type}

        return {}

    def extract_department(self, query: str) -> Dict:
        """Identifica departamento mencionado"""
        departments = {
            'rh|recursos humanos|hr': 'hr',
            'ti|tecnologia|tech': 'it',
            'financeiro|finance': 'finance',
            'vendas|sales': 'sales',
            'marketing': 'marketing',
            'legal|jurídico': 'legal'
        }

        query_lower = query.lower()
        for pattern, dept in departments.items():
            if re.search(pattern, query_lower):
                return {'department': dept}

        return {}

    def route_query(self, query: str, user_context: Dict = None) -> Dict:
        """
        Roteia query e determina filtros

        Returns:
            {
                'route': route_name,
                'index': index_name,
                'filters': metadata_filters,
                'enhanced_query': query_modificada
            }
        """
        # Extract all metadata
        all_filters = {}

        # Apply extractors
        all_filters.update(self.extract_date_filter(query))
        all_filters.update(self.extract_document_type(query))
        all_filters.update(self.extract_department(query))

        # Apply custom extractors
        for extractor in self.metadata_extractors:
            all_filters.update(extractor(query))

        # Add user context filters
        if user_context:
            if 'user_department' in user_context:
                all_filters.setdefault('department', user_context['user_department'])
            if 'access_level' in user_context:
                all_filters['min_access_level'] = user_context['access_level']

        # Find matching route
        selected_route = 'default'
        selected_index = 'main_index'

        for route_name, route_info in self.routes.items():
            if route_info['condition'](query, all_filters):
                selected_route = route_name
                selected_index = route_info['index']
                route_info['hits'] += 1
                break

        self.routing_stats[selected_route] += 1

        return {
            'route': selected_route,
            'index': selected_index,
            'filters': all_filters,
            'enhanced_query': query,
            'original_query': query
        }

# === EXEMPLO DE USO ===

print("\n🔍 Smart Query Router Demo\n")
print("="*70)

router = SmartQueryRouter()

# Registrar rotas
router.register_route(
    'technical_docs',
    lambda q, f: f.get('document_type') in ['documentation', 'code'],
    'technical_index'
)

router.register_route(
    'hr_policies',
    lambda q, f: f.get('department') == 'hr' or 'política' in q.lower(),
    'hr_index'
)

router.register_route(
    'recent_docs',
    lambda q, f: 'date_from' in f or 'últimos' in q.lower(),
    'time_series_index'
)

# Registrar extrator customizado
def extract_priority(query: str) -> Dict:
    if 'urgente' in query.lower() or 'prioritário' in query.lower():
        return {'priority': 'high'}
    return {}


if __name__ == "__main__":
    router.add_metadata_extractor(extract_priority)

    # Teste de queries
    test_queries = [
        ("Como configurar o servidor?", {'user_department': 'it'}),
        ("Política de férias do RH de 2024", {'user_department': 'hr'}),
        ("Relatórios de vendas dos últimos 30 dias", {'user_department': 'sales'}),
        ("Manual de Python código exemplo", None),
        ("Contrato urgente do cliente X", {'user_department': 'legal'}),
        ("Email sobre reunião em janeiro", None),
    ]

    results = []
    for query, user_ctx in test_queries:
        print(f"\n📝 Query: '{query}'")
        if user_ctx:
            print(f"   User: {user_ctx}")

        result = router.route_query(query, user_ctx)
        results.append(result)

        print(f"   ➡️  Route: {result['route']}")
        print(f"   📊 Index: {result['index']}")
        if result['filters']:
            print(f"   🔍 Filters: {result['filters']}")

    # === VISUALIZAÇÃO ===

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Route distribution
    ax = axes[0, 0]
    routes = list(router.routing_stats.keys())
    counts = list(router.routing_stats.values())

    ax.pie(counts, labels=routes, autopct='%1.1f%%', startangle=90)
    ax.set_title('Query Routing Distribution')

    # 2. Filters applied
    ax = axes[0, 1]
    filter_types = defaultdict(int)
    for result in results:
        for filter_key in result['filters'].keys():
            filter_types[filter_key] += 1

    if filter_types:
        ax.barh(list(filter_types.keys()), list(filter_types.values()),
                color='skyblue', alpha=0.7)
        ax.set_xlabel('Count')
        ax.set_title('Metadata Filters Applied')
        ax.grid(axis='x', alpha=0.3)

    # 3. Index usage
    ax = axes[1, 0]
    index_usage = defaultdict(int)
    for result in results:
        index_usage[result['index']] += 1

    ax.bar(range(len(index_usage)), list(index_usage.values()),
           color='lightgreen', alpha=0.7)
    ax.set_xticks(range(len(index_usage)))
    ax.set_xticklabels(list(index_usage.keys()), rotation=45, ha='right')
    ax.set_ylabel('Queries')
    ax.set_title('Index Usage')
    ax.grid(axis='y', alpha=0.3)

    # 4. Filter complexity (number of filters per query)
    ax = axes[1, 1]
    filter_counts = [len(r['filters']) for r in results]
    ax.hist(filter_counts, bins=range(0, max(filter_counts)+2), 
            color='coral', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Filters')
    ax.set_ylabel('Frequency')
    ax.set_title('Filter Complexity Distribution')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('go1704_query_routing.png', dpi=120, bbox_inches='tight')
    print(f"Grafico salvo: go1704_query_routing.png")
    plt.show()
    print("\n📊 Gráfico salvo: query_routing_analysis.png")

    # Summary
    print("\n" + "="*70)
    print("📈 ROUTING STATISTICS")
    print("="*70)
    print(f"\nTotal queries: {len(results)}")
    print(f"\nRoutes hit:")
    for route, count in router.routing_stats.items():
        print(f"  - {route}: {count} ({count/len(results)*100:.1f}%)")

    print(f"\nMost common filters:")
    for filter_type, count in sorted(filter_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {filter_type}: {count}")

    print("\n✅ Smart Query Router implementado!")
