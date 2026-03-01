# GO1709-21bGraphragKnowledgeGraphEnhanced
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import networkx as nx

class GraphRAG:
    """
    GraphRAG: Combines knowledge graphs with RAG

    Benefits:
    - Structured relationships between entities
    - Multi-hop reasoning
    - Better context understanding
    - Explicit knowledge representation

    Use cases:
    - Scientific papers (citations, concepts)
    - Legal documents (cases, laws, precedents)
    - Technical docs (APIs, dependencies)
    """

    def __init__(self):
        # Knowledge graph: (source, relation, target)
        self.graph = nx.DiGraph()

        # Document store
        self.documents = {}

        # Entity mentions in documents
        self.entity_to_docs = defaultdict(list)

    def add_document(self, doc_id: str, text: str):
        """Add document and extract entities/relations"""
        self.documents[doc_id] = text

        # Extract entities (simplified: capitalize words)
        entities = self._extract_entities(text)

        # Store entity-document mapping
        for entity in entities:
            self.entity_to_docs[entity].append(doc_id)

            # Add entity node to graph
            if not self.graph.has_node(entity):
                self.graph.add_node(entity, type='entity')

        # Extract relations (simplified)
        relations = self._extract_relations(text, entities)

        # Add relations to graph
        for source, relation, target in relations:
            self.graph.add_edge(source, target, relation=relation)

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities (simplified: capitalized words)"""
        words = text.split()
        entities = []

        for i, word in enumerate(words):
            # Multi-word entities
            if word[0].isupper() and len(word) > 2:
                # Check if next word is also capitalized
                if i + 1 < len(words) and words[i+1][0].isupper():
                    entity = f"{word} {words[i+1]}"
                else:
                    entity = word

                entities.append(entity.strip('.,!?'))

        return list(set(entities))

    def _extract_relations(self, text: str, entities: List[str]) -> List[Tuple[str, str, str]]:
        """Extract relations between entities"""
        relations = []

        # Simple patterns
        patterns = [
            ('is a', 'IS_A'),
            ('uses', 'USES'),
            ('extends', 'EXTENDS'),
            ('implements', 'IMPLEMENTS'),
            ('related to', 'RELATED_TO'),
            ('part of', 'PART_OF'),
        ]

        text_lower = text.lower()

        for e1 in entities:
            for e2 in entities:
                if e1 != e2:
                    # Check if both entities appear in text
                    if e1.lower() in text_lower and e2.lower() in text_lower:
                        # Check for relation patterns
                        for pattern, rel_type in patterns:
                            # Simple: if "E1 pattern E2" in text
                            pattern_text = f"{e1.lower()} {pattern} {e2.lower()}"
                            if pattern in text_lower:
                                # Verify order
                                e1_pos = text_lower.find(e1.lower())
                                e2_pos = text_lower.find(e2.lower())
                                pattern_pos = text_lower.find(pattern)

                                if e1_pos < pattern_pos < e2_pos:
                                    relations.append((e1, rel_type, e2))

        return relations

    def query_with_graph(self, query: str, k: int = 5, hops: int = 2) -> List[Dict]:
        """
        Query using graph structure

        Steps:
        1. Extract query entities
        2. Find related entities in graph (multi-hop)
        3. Retrieve documents mentioning related entities
        4. Rank by relevance
        """
        # Extract query entities
        query_entities = self._extract_entities(query)

        if not query_entities:
            return []

        # Find related entities via graph traversal
        related_entities = set(query_entities)

        for entity in query_entities:
            if entity in self.graph:
                # Get neighbors within k hops
                neighbors = self._get_k_hop_neighbors(entity, hops)
                related_entities.update(neighbors)

        # Retrieve documents mentioning related entities
        doc_scores = defaultdict(float)

        for entity in related_entities:
            docs = self.entity_to_docs.get(entity, [])

            # Score based on:
            # 1. Query entity match (1.0)
            # 2. 1-hop neighbor (0.5)
            # 3. 2-hop neighbor (0.25)
            if entity in query_entities:
                score = 1.0
            else:
                # Calculate distance from query entities
                min_distance = min(
                    nx.shortest_path_length(self.graph, qe, entity)
                    if nx.has_path(self.graph, qe, entity) else float('inf')
                    for qe in query_entities if qe in self.graph
                )
                score = 0.5 ** min_distance if min_distance < float('inf') else 0.1

            for doc_id in docs:
                doc_scores[doc_id] += score

        # Sort and return top k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return [
            {
                'doc_id': doc_id,
                'text': self.documents[doc_id],
                'score': score,
                'related_entities': list(related_entities)
            }
            for doc_id, score in sorted_docs
        ]

    def _get_k_hop_neighbors(self, entity: str, k: int) -> Set[str]:
        """Get all entities within k hops"""
        if entity not in self.graph:
            return set()

        neighbors = set([entity])
        current_level = {entity}

        for _ in range(k):
            next_level = set()
            for node in current_level:
                # Add successors and predecessors
                next_level.update(self.graph.successors(node))
                next_level.update(self.graph.predecessors(node))

            neighbors.update(next_level)
            current_level = next_level

        return neighbors

    def visualize_graph(self, highlight_entities: List[str] = None):
        """Visualize knowledge graph"""
        plt.figure(figsize=(12, 8))

        pos = nx.spring_layout(self.graph, k=0.5, iterations=50)

        # Draw nodes
        node_colors = []
        for node in self.graph.nodes():
            if highlight_entities and node in highlight_entities:
                node_colors.append('lightcoral')
            else:
                node_colors.append('skyblue')

        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, 
                              node_size=500, alpha=0.9)

        # Draw edges with labels
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5, arrows=True, 
                              arrowsize=20, edge_color='gray')

        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=8, font_weight='bold')

        # Draw edge labels
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=6)

        plt.title("Knowledge Graph")
        plt.axis('off')
        plt.tight_layout()
        return plt

# === DEMO ===

print("📐 GraphRAG Demo\n")
print("="*70)

graph_rag = GraphRAG()

# Add documents with entities and relations
documents = [
    ("doc1", "Machine Learning is a subset of Artificial Intelligence. ML uses algorithms to learn from data."),
    ("doc2", "Deep Learning is a subset of Machine Learning. Deep Learning uses Neural Networks."),
    ("doc3", "Neural Networks are computational models inspired by biological neurons."),
    ("doc4", "Transformers are a type of Neural Networks. Transformers use attention mechanisms."),
    ("doc5", "BERT is a Transformer model. BERT uses masked language modeling."),
]

for doc_id, text in documents:
    graph_rag.add_document(doc_id, text)
    print(f"✅ Added {doc_id}")

print(f"\n📊 Graph Statistics:")
print(f"   Nodes: {graph_rag.graph.number_of_nodes()}")
print(f"   Edges: {graph_rag.graph.number_of_edges()}")

# Test queries
test_queries = [
    "What is Deep Learning?",
    "Explain Transformers",
    "How do Neural Networks work?"
]

for query in test_queries:
    print(f"\n📌 Query: '{query}'")
    print("-"*70)

    results = graph_rag.query_with_graph(query, k=3, hops=2)

    print(f"Found {len(results)} documents:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. [{result['doc_id']}] (score={result['score']:.2f})")
        print(f"   Text: {result['text'][:80]}...")
        print(f"   Related entities: {', '.join(result['related_entities'][:5])}")

# Visualize graph
print("\n📊 Visualizing knowledge graph...")
query_entities = graph_rag._extract_entities("Deep Learning Transformers")
plt = graph_rag.visualize_graph(highlight_entities=query_entities)
plt.savefig('knowledge_graph.png', dpi=150, bbox_inches='tight')
print("✅ Graph saved: knowledge_graph.png")

print("\n✅ GraphRAG implementado!")
print("\n💡 ADVANTAGES:")
print("   - Multi-hop reasoning")
print("   - Explicit knowledge structure")
print("   - Better for complex queries")
print("   - Handles relationships well")
print("\n⚠️  CHALLENGES:")
print("   - Entity extraction quality critical")
print("   - Relation extraction is hard")
print("   - Graph construction overhead")
print("   - Scalability for large graphs")
