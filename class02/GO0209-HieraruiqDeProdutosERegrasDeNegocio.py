# GO0209-HieraruiqDeProdutosERegrasDeNegócio
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import RDFS, OWL
import matplotlib.pyplot as plt
import networkx as nx

# ═══════════════════════════════════════════════════════════════════
# 1. ONTOLOGIA E-COMMERCE COM RDF/OWL
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("ONTOLOGIA E-COMMERCE - RDF/OWL")
print("="*70)

# Criar grafo RDF
g = Graph()

# Definir namespace
EX = Namespace("http://example.org/ecommerce#")
g.bind("ex", EX)
g.bind("owl", OWL)

print("\n🏗️ CONSTRUINDO ONTOLOGIA...")

# ═══════════════════════════════════════════════════════════════════
# 2. CLASSES (HIERARQUIA DE PRODUTOS)
# ═══════════════════════════════════════════════════════════════════

# Classe raiz
g.add((EX.Produto, RDF.type, OWL.Class))

# Subclasses de Produto
g.add((EX.Eletronico, RDF.type, OWL.Class))
g.add((EX.Eletronico, RDFS.subClassOf, EX.Produto))

g.add((EX.Roupa, RDF.type, OWL.Class))
g.add((EX.Roupa, RDFS.subClassOf, EX.Produto))

g.add((EX.Livro, RDF.type, OWL.Class))
g.add((EX.Livro, RDFS.subClassOf, EX.Produto))

# Subclasses de Eletrônico
g.add((EX.Smartphone, RDF.type, OWL.Class))
g.add((EX.Smartphone, RDFS.subClassOf, EX.Eletronico))

g.add((EX.Laptop, RDF.type, OWL.Class))
g.add((EX.Laptop, RDFS.subClassOf, EX.Eletronico))

g.add((EX.Tablet, RDF.type, OWL.Class))
g.add((EX.Tablet, RDFS.subClassOf, EX.Eletronico))

# Subclasses de Roupa
g.add((EX.Camiseta, RDF.type, OWL.Class))
g.add((EX.Camiseta, RDFS.subClassOf, EX.Roupa))

g.add((EX.Calca, RDF.type, OWL.Class))
g.add((EX.Calca, RDFS.subClassOf, EX.Roupa))

print(f"✅ Classes criadas: 9 classes na hierarquia")

# ═══════════════════════════════════════════════════════════════════
# 3. PROPRIEDADES (OBJECT PROPERTIES E DATATYPE PROPERTIES)
# ═══════════════════════════════════════════════════════════════════

# Object Properties
g.add((EX.temCategoria, RDF.type, OWL.ObjectProperty))
g.add((EX.fabricadoPor, RDF.type, OWL.ObjectProperty))
g.add((EX.vendidoPor, RDF.type, OWL.ObjectProperty))

# Datatype Properties
g.add((EX.preco, RDF.type, OWL.DatatypeProperty))
g.add((EX.estoque, RDF.type, OWL.DatatypeProperty))
g.add((EX.peso, RDF.type, OWL.DatatypeProperty))
g.add((EX.marca, RDF.type, OWL.DatatypeProperty))
g.add((EX.cor, RDF.type, OWL.DatatypeProperty))
g.add((EX.tamanho, RDF.type, OWL.DatatypeProperty))

print(f"✅ Propriedades criadas: 9 propriedades")

# ═══════════════════════════════════════════════════════════════════
# 4. INSTÂNCIAS (INDIVÍDUOS)
# ═══════════════════════════════════════════════════════════════════

# Smartphones
g.add((EX.iPhone15, RDF.type, EX.Smartphone))
g.add((EX.iPhone15, EX.marca, Literal("Apple")))
g.add((EX.iPhone15, EX.preco, Literal(4999.00)))
g.add((EX.iPhone15, EX.estoque, Literal(50)))
g.add((EX.iPhone15, EX.peso, Literal(171)))
g.add((EX.iPhone15, EX.cor, Literal("Preto")))

g.add((EX.GalaxyS24, RDF.type, EX.Smartphone))
g.add((EX.GalaxyS24, EX.marca, Literal("Samsung")))
g.add((EX.GalaxyS24, EX.preco, Literal(4499.00)))
g.add((EX.GalaxyS24, EX.estoque, Literal(75)))
g.add((EX.GalaxyS24, EX.peso, Literal(168)))
g.add((EX.GalaxyS24, EX.cor, Literal("Branco")))

# Laptops
g.add((EX.MacBookPro, RDF.type, EX.Laptop))
g.add((EX.MacBookPro, EX.marca, Literal("Apple")))
g.add((EX.MacBookPro, EX.preco, Literal(12999.00)))
g.add((EX.MacBookPro, EX.estoque, Literal(20)))
g.add((EX.MacBookPro, EX.peso, Literal(1600)))

g.add((EX.DellXPS, RDF.type, EX.Laptop))
g.add((EX.DellXPS, EX.marca, Literal("Dell")))
g.add((EX.DellXPS, EX.preco, Literal(8999.00)))
g.add((EX.DellXPS, EX.estoque, Literal(30)))
g.add((EX.DellXPS, EX.peso, Literal(1800)))

# Roupas
g.add((EX.CamisetaBranca, RDF.type, EX.Camiseta))
g.add((EX.CamisetaBranca, EX.marca, Literal("Nike")))
g.add((EX.CamisetaBranca, EX.preco, Literal(89.90)))
g.add((EX.CamisetaBranca, EX.estoque, Literal(100)))
g.add((EX.CamisetaBranca, EX.cor, Literal("Branco")))
g.add((EX.CamisetaBranca, EX.tamanho, Literal("M")))

g.add((EX.CalcaJeans, RDF.type, EX.Calca))
g.add((EX.CalcaJeans, EX.marca, Literal("Levi's")))
g.add((EX.CalcaJeans, EX.preco, Literal(299.00)))
g.add((EX.CalcaJeans, EX.estoque, Literal(60)))
g.add((EX.CalcaJeans, EX.cor, Literal("Azul")))
g.add((EX.CalcaJeans, EX.tamanho, Literal("38")))

# Livros
g.add((EX.LivroIA, RDF.type, EX.Livro))
g.add((EX.LivroIA, EX.marca, Literal("Elsevier")))
g.add((EX.LivroIA, EX.preco, Literal(120.00)))
g.add((EX.LivroIA, EX.estoque, Literal(40)))
g.add((EX.LivroIA, EX.peso, Literal(800)))

print(f"✅ Instâncias criadas: 7 produtos")
print(f"   Total de triplas no grafo: {len(g)}")

# ═══════════════════════════════════════════════════════════════════
# 5. CONSULTAS SPARQL
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("CONSULTAS SPARQL")
print("="*70)

# Consulta 1: Todos os smartphones
print("\n🔍 Consulta 1: Listar todos os smartphones com preços")
query1 = """
    PREFIX ex: <http://example.org/ecommerce#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT ?produto ?marca ?preco ?estoque
    WHERE {
        ?produto rdf:type ex:Smartphone .
        ?produto ex:marca ?marca .
        ?produto ex:preco ?preco .
        ?produto ex:estoque ?estoque .
    }
"""

resultados1 = g.query(query1)
for row in resultados1:
    produto = str(row.produto).split('#')[1]
    print(f"   • {produto}: {row.marca} - R$ {row.preco} (estoque: {row.estoque})")

# Consulta 2: Produtos com preço < 500
print("\n🔍 Consulta 2: Produtos abaixo de R$ 500")
query2 = """
    PREFIX ex: <http://example.org/ecommerce#>

    SELECT ?produto ?preco
    WHERE {
        ?produto ex:preco ?preco .
        FILTER (?preco < 500)
    }
    ORDER BY ?preco
"""

resultados2 = g.query(query2)
for row in resultados2:
    produto = str(row.produto).split('#')[1]
    print(f"   • {produto}: R$ {row.preco}")

# Consulta 3: Produtos eletrônicos (usando inferência de subclasse)
print("\n🔍 Consulta 3: Todos os eletrônicos (smartphones + laptops)")
query3 = """
    PREFIX ex: <http://example.org/ecommerce#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?produto ?tipo ?marca ?preco
    WHERE {
        ?produto rdf:type ?tipo .
        ?tipo rdfs:subClassOf* ex:Eletronico .
        ?produto ex:marca ?marca .
        ?produto ex:preco ?preco .
    }
    ORDER BY DESC(?preco)
"""

resultados3 = g.query(query3)
for row in resultados3:
    produto = str(row.produto).split('#')[1]
    tipo = str(row.tipo).split('#')[1]
    print(f"   • {produto} ({tipo}): {row.marca} - R$ {row.preco}")

# Consulta 4: Produtos por marca
print("\n🔍 Consulta 4: Produtos da Apple")
query4 = """
    PREFIX ex: <http://example.org/ecommerce#>

    SELECT ?produto ?preco
    WHERE {
        ?produto ex:marca "Apple" .
        ?produto ex:preco ?preco .
    }
"""

resultados4 = g.query(query4)
for row in resultados4:
    produto = str(row.produto).split('#')[1]
    print(f"   • {produto}: R$ {row.preco}")

# ═══════════════════════════════════════════════════════════════════
# 6. ESTATÍSTICAS E ANÁLISES
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("ESTATÍSTICAS DA ONTOLOGIA")
print("="*70)

# Contar classes
classes_query = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT (COUNT(?class) as ?total)
    WHERE {
        ?class rdf:type owl:Class .
    }
"""
result = list(g.query(classes_query))[0]
print(f"\n📊 Classes: {result.total}")

# Contar propriedades
props_query = """
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

    SELECT (COUNT(?prop) as ?total)
    WHERE {
        {?prop rdf:type owl:ObjectProperty}
        UNION
        {?prop rdf:type owl:DatatypeProperty}
    }
"""
result = list(g.query(props_query))[0]
print(f"📊 Propriedades: {result.total}")

# Contar instâncias
instances_query = """
    PREFIX ex: <http://example.org/ecommerce#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT (COUNT(DISTINCT ?instance) as ?total)
    WHERE {
        ?instance rdf:type ?type .
        ?type rdfs:subClassOf* ex:Produto .
    }
"""
result = list(g.query(instances_query))[0]
print(f"📊 Instâncias de produtos: {result.total}")

# Estatísticas de preço
precos = []
query_precos = """
    PREFIX ex: <http://example.org/ecommerce#>

    SELECT ?preco
    WHERE {
        ?produto ex:preco ?preco .
    }
"""
for row in g.query(query_precos):
    precos.append(float(row.preco))

print(f"\n💰 ANÁLISE DE PREÇOS:")
print(f"   Média: R$ {sum(precos)/len(precos):.2f}")
print(f"   Mínimo: R$ {min(precos):.2f}")
print(f"   Máximo: R$ {max(precos):.2f}")

# ═══════════════════════════════════════════════════════════════════
# 7. EXPORTAR ONTOLOGIA
# ═══════════════════════════════════════════════════════════════════

print("\n📝 EXPORTANDO ONTOLOGIA...")

# Salvar em diferentes formatos
g.serialize(destination='ecommerce_ontology.rdf', format='xml')
print("   ✅ Salvo: ecommerce_ontology.rdf (RDF/XML)")

g.serialize(destination='ecommerce_ontology.ttl', format='turtle')
print("   ✅ Salvo: ecommerce_ontology.ttl (Turtle)")

# ═══════════════════════════════════════════════════════════════════
# 8. VISUALIZAÇÃO DA HIERARQUIA
# ═══════════════════════════════════════════════════════════════════

print("\n📊 Gerando visualização da hierarquia de classes...")

# Criar grafo NetworkX para visualização
G_viz = nx.DiGraph()

# Adicionar hierarquia de classes
hierarchy_query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?subclass ?superclass
    WHERE {
        ?subclass rdfs:subClassOf ?superclass .
    }
"""

for row in g.query(hierarchy_query):
    subclass = str(row.subclass).split('#')[1]
    superclass = str(row.superclass).split('#')[1]
    G_viz.add_edge(superclass, subclass)

# Visualizar
plt.figure(figsize=(14, 10))
pos = nx.spring_layout(G_viz, k=2, iterations=50)

nx.draw_networkx_nodes(G_viz, pos, node_size=3000, node_color='lightblue', alpha=0.9)
nx.draw_networkx_edges(G_viz, pos, arrows=True, arrowsize=20, alpha=0.6, width=2)
nx.draw_networkx_labels(G_viz, pos, font_size=10, font_weight='bold')

plt.title("Hierarquia de Classes - Ontologia E-commerce", fontsize=14, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.show()

# ═══════════════════════════════════════════════════════════════════
# 9. RELATÓRIO FINAL
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("RELATÓRIO FINAL - ONTOLOGIA E-COMMERCE")
print("="*70)

print(f"\n🏗️ ESTRUTURA:")
print(f"   • Padrão: RDF/OWL")
print(f"   • Total de triplas: {len(g)}")
print(f"   • Classes: 9 (hierarquia de produtos)")
print(f"   • Propriedades: 9 (dados + objetos)")
print(f"   • Instâncias: 7 produtos")

print(f"\n🔍 FUNCIONALIDADES:")
print(f"   ✅ Consultas SPARQL")
print(f"   ✅ Inferência de subclasses")
print(f"   ✅ Filtros e ordenação")
print(f"   ✅ Exportação RDF/XML e Turtle")

print(f"\n🎯 APLICAÇÕES REAIS:")
print(f"   • E-commerce: Catalogação de produtos")
print(f"   • Recomendação: Produtos similares (mesma hierarquia)")
print(f"   • Busca semântica: Entender relações")
print(f"   • Integração: Padrão W3C interoperável")

print(f"\n💡 VANTAGENS DE ONTOLOGIAS:")
print(f"   ✅ PADRÃO: RDF/OWL é universal (W3C)")
print(f"   ✅ INFERÊNCIA: Raciocínio automático")
print(f"   ✅ HIERARQUIA: Relações is-a explícitas")
print(f"   ✅ SPARQL: Linguagem de consulta poderosa")
print(f"   ✅ INTEROPERABILIDADE: Fácil integração")

print("\n✅ ONTOLOGIA E-COMMERCE COMPLETA!")
