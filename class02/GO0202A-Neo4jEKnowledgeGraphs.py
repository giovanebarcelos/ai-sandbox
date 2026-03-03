# GO0202A-16aNeo4jEKnowledgeGraphs
from neo4j import GraphDatabase
import json

print("🗄️ NEO4J + PYTHON - KNOWLEDGE GRAPH")
print("=" * 70)

# Conectar ao Neo4j
class KnowledgeGraphDB:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def criar_pessoa(self, nome, idade, profissao):
        with self.driver.session() as session:
            result = session.run(
                """
                CREATE (p:Pessoa {nome: $nome, idade: $idade, profissao: $profissao})
                RETURN p
                """,
                nome=nome, idade=idade, profissao=profissao
            )
            return result.single()[0]

    def criar_amizade(self, pessoa1, pessoa2, desde):
        with self.driver.session() as session:
            session.run(
                """
                MATCH (p1:Pessoa {nome: $pessoa1}), (p2:Pessoa {nome: $pessoa2})
                CREATE (p1)-[:AMIGO_DE {desde: $desde}]->(p2)
                """,
                pessoa1=pessoa1, pessoa2=pessoa2, desde=desde
            )

    def buscar_amigos(self, nome):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (p:Pessoa {nome: $nome})-[:AMIGO_DE]->(amigo)
                RETURN amigo.nome AS nome, amigo.idade AS idade
                """,
                nome=nome
            )
            return [dict(record) for record in result]

    def caminho_mais_curto(self, pessoa1, pessoa2):
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = shortestPath(
                    (p1:Pessoa {nome: $pessoa1})-[:AMIGO_DE*]-(p2:Pessoa {nome: $pessoa2})
                )
                RETURN [node in nodes(path) | node.nome] AS caminho, 
                       length(path) AS distancia
                """,
                pessoa1=pessoa1, pessoa2=pessoa2
            )
            record = result.single()
            if record:
                return record['caminho'], record['distancia']
            return None, None

    def recomendar_amigos(self, nome, limite=5):
        """Amigos de amigos com interesses comuns"""
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (eu:Pessoa {nome: $nome})-[:AMIGO_DE]->()-[:AMIGO_DE]->(recomendado)
                WHERE NOT (eu)-[:AMIGO_DE]->(recomendado)
                  AND recomendado <> eu
                RETURN recomendado.nome AS nome, COUNT(*) AS conexoes_mutuas
                ORDER BY conexoes_mutuas DESC
                LIMIT $limite
                """,
                nome=nome, limite=limite
            )
            return [dict(record) for record in result]

# Exemplo de uso (requer Neo4j rodando)
try:
    # Conectar (ajustar credenciais)
    db = KnowledgeGraphDB(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )

    print("✅ Conectado ao Neo4j\n")

    # Criar pessoas
    print("📝 Criando pessoas...")
    db.criar_pessoa("João", 30, "Engenheiro")
    db.criar_pessoa("Maria", 28, "Designer")
    db.criar_pessoa("Pedro", 32, "Médico")
    db.criar_pessoa("Ana", 27, "Professora")

    # Criar amizades
    print("🤝 Criando conexões...")
    db.criar_amizade("João", "Maria", 2020)
    db.criar_amizade("Maria", "Pedro", 2019)
    db.criar_amizade("Pedro", "Ana", 2021)
    db.criar_amizade("João", "Pedro", 2018)

    # Buscar amigos
    print("\n🔍 Amigos de João:")
    amigos = db.buscar_amigos("João")
    for amigo in amigos:
        print(f"   • {amigo['nome']} ({amigo['idade']} anos)")

    # Caminho mais curto
    print("\n🛤️ Caminho mais curto entre João e Ana:")
    caminho, distancia = db.caminho_mais_curto("João", "Ana")
    if caminho:
        print(f"   Caminho: {' → '.join(caminho)}")
        print(f"   Distância: {distancia} conexões")

    # Recomendações
    print("\n💡 Recomendações de amigos para João:")
    recomendacoes = db.recomendar_amigos("João", limite=3)
    for rec in recomendacoes:
        print(f"   • {rec['nome']} ({rec['conexoes_mutuas']} amigos em comum)")

    db.close()
    print("\n✅ Conexão fechada")

except Exception as e:
    print(f"❌ Erro: {e}")
    print("\n💡 Para rodar este exemplo:")
    print("   1. Instale Neo4j Desktop: https://neo4j.com/download/")
    print("   2. Crie um banco local")
    print("   3. Ajuste credenciais no código")
    print("   4. Execute novamente")
