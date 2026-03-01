# GO0709-Etapa4EstratégiasDeMarketing
# ═══════════════════════════════════════════════════════════════════
# ETAPA 4: ESTRATÉGIAS DE MARKETING
# ═══════════════════════════════════════════════════════════════════

print("="*70)
print("ESTRATÉGIAS DE MARKETING POR SEGMENTO")
print("="*70)

estrategias = {
    "Jovens Econômicos": """
    🎯 PERFIL: Jovens (20-30), renda baixa, gasto baixo

    📢 ESTRATÉGIA:
       • Oferecer cupons de desconto e promoções agressivas
       • Produtos de entrada (baixo custo)
       • Programa de fidelidade com pontos
       • Marketing em redes sociais
       • Parcelamento sem juros

    💰 POTENCIAL: Médio (futuro - quando renda aumentar)
    """,

    "Profissionais Premium": """
    🎯 PERFIL: Meia-idade (40-50), alta renda, alto gasto

    📢 ESTRATÉGIA:
       • Produtos premium e exclusivos
       • Atendimento VIP personalizado
       • Lançamentos em primeira mão
       • Eventos exclusivos
       • Programa de cashback generoso

    💰 POTENCIAL: MUITO ALTO (clientes ideais!)
    """,

    "Maduros Conservadores": """
    🎯 PERFIL: Idosos (55-65), alta renda, gasto baixo

    📢 ESTRATÉGIA:
       • Foco em confiabilidade e qualidade
       • Marketing educacional
       • Garantias estendidas
       • Demonstrações detalhadas de produtos
       • Ênfase em segurança e suporte

    💰 POTENCIAL: Médio-Alto (convencer a gastar mais)
    """,

    "Jovens Gastadores": """
    🎯 PERFIL: Jovens (25-35), renda média, gasto alto

    📢 ESTRATÉGIA:
       • Tendências e novidades
       • Marketing de influência
       • Edições limitadas
       • Experiências e eventos
       • Programa de indicação com recompensas

    💰 POTENCIAL: ALTO (gastam além da renda!)
    """
}

for nome, estrategia in estrategias.items():
    print(f"\n{nome}")
    print("─" * 70)
    print(estrategia)

# ───────────────────────────────────────────────────────────────────
# COMPARAR COM DBSCAN
# ───────────────────────────────────────────────────────────────────

from sklearn.cluster import DBSCAN

print("\n" + "="*70)
print("COMPARAÇÃO: K-MEANS vs DBSCAN")
print("="*70)

# Aplicar DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

n_clusters_dbscan = len(set(df['DBSCAN_Cluster'])) - (1 if -1 in df['DBSCAN_Cluster'].values else 0)
n_noise = list(df['DBSCAN_Cluster']).count(-1)

print(f"\nK-Means:  {df['Cluster'].nunique()} clusters")
print(f"DBSCAN:   {n_clusters_dbscan} clusters + {n_noise} outliers")

# Visualizar comparação
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# K-Means
axes[0].scatter(df['Annual_Income'], df['Spending_Score'], 
               c=df['Cluster'], cmap='viridis', s=50, alpha=0.6, 
               edgecolors='k')
axes[0].set_xlabel('Renda Anual (mil)', fontsize=12)
axes[0].set_ylabel('Spending Score', fontsize=12)
axes[0].set_title(f'K-Means (K=4)', fontsize=14)

# DBSCAN
scatter = axes[1].scatter(df['Annual_Income'], df['Spending_Score'], 
                         c=df['DBSCAN_Cluster'], cmap='viridis', 
                         s=50, alpha=0.6, edgecolors='k')
axes[1].set_xlabel('Renda Anual (mil)', fontsize=12)
axes[1].set_ylabel('Spending Score', fontsize=12)
axes[1].set_title(f'DBSCAN ({n_clusters_dbscan} clusters, {n_noise} outliers)', 
                 fontsize=14)

plt.tight_layout()
plt.show()

print("\n📊 ANÁLISE:")
print("  • K-Means: Clusters mais uniformes, tamanhos similares")
print("  • DBSCAN: Detecta outliers, clusters baseados em densidade")
print("  • Para este caso: K-Means é mais adequado (grupos bem definidos)")

# ───────────────────────────────────────────────────────────────────
# EXPORTAR RESULTADOS
# ───────────────────────────────────────────────────────────────────

# Salvar para análise posterior
df.to_csv('customer_segments.csv', index=False)

print("\n✅ ATIVIDADE COMPLETA!")
print("📁 Arquivo 'customer_segments.csv' salvo com sucesso!")

print("\n" + "="*70)
print("🎓 PARABÉNS! Você completou a segmentação de clientes!")
print("="*70)
