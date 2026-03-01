# GO1707-19aProductionMonitoringObservability
import time
from typing import Dict, List
from datetime import datetime, timedelta
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

class RAGMonitor:
    """
    Production monitoring for RAG systems

    Tracks:
    - Query latency (p50, p95, p99)
    - Retrieval quality (relevance scores)
    - Generation quality (token counts, errors)
    - Cost metrics (tokens used, API calls)
    - Error rates and types
    - User satisfaction (if available)
    """

    def __init__(self):
        self.metrics = {
            'queries': [],
            'latencies': [],
            'retrieval_scores': [],
            'token_counts': [],
            'errors': [],
            'costs': []
        }

        # Aggregated stats
        self.hourly_stats = defaultdict(lambda: {
            'count': 0,
            'total_latency': 0,
            'total_tokens': 0,
            'total_cost': 0,
            'errors': 0
        })

        print("✅ RAG Monitor initialized")

    def log_query(self, 
                  query: str,
                  latency: float,
                  retrieval_score: float,
                  token_count: int,
                  cost: float,
                  error: str = None):
        """Log a query execution"""
        timestamp = datetime.now()
        hour_key = timestamp.strftime('%Y-%m-%d %H:00')

        # Store detailed metrics
        self.metrics['queries'].append({
            'timestamp': timestamp,
            'query': query,
            'latency': latency,
            'retrieval_score': retrieval_score,
            'token_count': token_count,
            'cost': cost,
            'error': error
        })

        self.metrics['latencies'].append(latency)
        self.metrics['retrieval_scores'].append(retrieval_score)
        self.metrics['token_counts'].append(token_count)
        self.metrics['costs'].append(cost)

        if error:
            self.metrics['errors'].append({
                'timestamp': timestamp,
                'error': error,
                'query': query
            })

        # Update hourly aggregations
        self.hourly_stats[hour_key]['count'] += 1
        self.hourly_stats[hour_key]['total_latency'] += latency
        self.hourly_stats[hour_key]['total_tokens'] += token_count
        self.hourly_stats[hour_key]['total_cost'] += cost
        if error:
            self.hourly_stats[hour_key]['errors'] += 1

    def get_statistics(self) -> Dict:
        """Get current statistics"""
        if not self.metrics['latencies']:
            return {}

        latencies = np.array(self.metrics['latencies'])

        return {
            'total_queries': len(self.metrics['queries']),
            'latency_p50': np.percentile(latencies, 50),
            'latency_p95': np.percentile(latencies, 95),
            'latency_p99': np.percentile(latencies, 99),
            'avg_retrieval_score': np.mean(self.metrics['retrieval_scores']),
            'total_tokens': sum(self.metrics['token_counts']),
            'total_cost': sum(self.metrics['costs']),
            'error_rate': len(self.metrics['errors']) / len(self.metrics['queries']),
            'queries_per_hour': len(self.metrics['queries']) / max(1, len(self.hourly_stats))
        }

    def get_alerts(self) -> List[Dict]:
        """Check for alert conditions"""
        alerts = []
        stats = self.get_statistics()

        if not stats:
            return alerts

        # Latency alert
        if stats['latency_p95'] > 5.0:
            alerts.append({
                'severity': 'warning',
                'metric': 'latency',
                'message': f"P95 latency high: {stats['latency_p95']:.2f}s (threshold: 5s)"
            })

        # Error rate alert
        if stats['error_rate'] > 0.05:
            alerts.append({
                'severity': 'critical',
                'metric': 'errors',
                'message': f"Error rate high: {stats['error_rate']*100:.1f}% (threshold: 5%)"
            })

        # Cost alert (monthly projection)
        monthly_cost_projection = stats['total_cost'] * 30 * 24 / max(1, len(self.hourly_stats))
        if monthly_cost_projection > 1000:
            alerts.append({
                'severity': 'warning',
                'metric': 'cost',
                'message': f"Projected monthly cost: ${monthly_cost_projection:.2f} (threshold: $1000)"
            })

        # Low quality alert
        if stats['avg_retrieval_score'] < 0.7:
            alerts.append({
                'severity': 'warning',
                'metric': 'quality',
                'message': f"Low retrieval quality: {stats['avg_retrieval_score']:.2f} (threshold: 0.7)"
            })

        return alerts

    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        stats = self.get_statistics()

        if not stats:
            return ""

        metrics = f"""
# HELP rag_queries_total Total number of RAG queries
# TYPE rag_queries_total counter
rag_queries_total {stats['total_queries']}

# HELP rag_latency_seconds Query latency in seconds
# TYPE rag_latency_seconds summary
rag_latency_seconds{{quantile="0.5"}} {stats['latency_p50']}
rag_latency_seconds{{quantile="0.95"}} {stats['latency_p95']}
rag_latency_seconds{{quantile="0.99"}} {stats['latency_p99']}

# HELP rag_retrieval_score Retrieval quality score
# TYPE rag_retrieval_score gauge
rag_retrieval_score {stats['avg_retrieval_score']}

# HELP rag_cost_dollars Total cost in dollars
# TYPE rag_cost_dollars counter
rag_cost_dollars {stats['total_cost']}

# HELP rag_errors_total Total number of errors
# TYPE rag_errors_total counter
rag_errors_total {len(self.metrics['errors'])}

# HELP rag_error_rate Error rate (0-1)
# TYPE rag_error_rate gauge
rag_error_rate {stats['error_rate']}
"""
        return metrics.strip()

    def visualize_dashboard(self):
        """Generate monitoring dashboard"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))

        # 1. Latency over time
        ax = axes[0, 0]
        if self.metrics['queries']:
            timestamps = [q['timestamp'] for q in self.metrics['queries']]
            latencies = [q['latency'] for q in self.metrics['queries']]

            ax.plot(timestamps, latencies, alpha=0.5, marker='o', markersize=3)
            ax.axhline(y=np.percentile(latencies, 95), color='red', 
                      linestyle='--', label='P95')
            ax.set_xlabel('Time')
            ax.set_ylabel('Latency (s)')
            ax.set_title('Query Latency Over Time')
            ax.legend()
            ax.grid(alpha=0.3)

        # 2. Latency distribution
        ax = axes[0, 1]
        if self.metrics['latencies']:
            ax.hist(self.metrics['latencies'], bins=20, color='skyblue', 
                   alpha=0.7, edgecolor='black')
            ax.axvline(np.percentile(self.metrics['latencies'], 95), 
                      color='red', linestyle='--', label='P95')
            ax.set_xlabel('Latency (s)')
            ax.set_ylabel('Frequency')
            ax.set_title('Latency Distribution')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

        # 3. Queries per hour
        ax = axes[1, 0]
        if self.hourly_stats:
            hours = sorted(self.hourly_stats.keys())
            counts = [self.hourly_stats[h]['count'] for h in hours]

            ax.bar(range(len(hours)), counts, color='lightgreen', alpha=0.7)
            ax.set_xlabel('Hour')
            ax.set_ylabel('Query Count')
            ax.set_title('Queries Per Hour')
            ax.set_xticks(range(len(hours)))
            ax.set_xticklabels([h.split()[1] for h in hours], rotation=45)
            ax.grid(axis='y', alpha=0.3)

        # 4. Cost over time
        ax = axes[1, 1]
        if self.metrics['queries']:
            timestamps = [q['timestamp'] for q in self.metrics['queries']]
            cumulative_costs = np.cumsum([q['cost'] for q in self.metrics['queries']])

            ax.plot(timestamps, cumulative_costs, linewidth=2, color='green')
            ax.set_xlabel('Time')
            ax.set_ylabel('Cumulative Cost ($)')
            ax.set_title('Cost Over Time')
            ax.grid(alpha=0.3)

        # 5. Retrieval quality
        ax = axes[2, 0]
        if self.metrics['retrieval_scores']:
            scores = self.metrics['retrieval_scores']
            ax.hist(scores, bins=20, color='purple', alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(scores), color='red', linestyle='--', label=f'Mean: {np.mean(scores):.2f}')
            ax.axvline(0.7, color='orange', linestyle='--', label='Threshold: 0.7')
            ax.set_xlabel('Retrieval Score')
            ax.set_ylabel('Frequency')
            ax.set_title('Retrieval Quality Distribution')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

        # 6. Error rate over time
        ax = axes[2, 1]
        if self.hourly_stats:
            hours = sorted(self.hourly_stats.keys())
            error_rates = [
                self.hourly_stats[h]['errors'] / max(self.hourly_stats[h]['count'], 1) * 100
                for h in hours
            ]

            ax.plot(range(len(hours)), error_rates, marker='o', 
                   linewidth=2, color='red', markersize=6)
            ax.axhline(y=5, color='orange', linestyle='--', 
                      alpha=0.5, label='Threshold: 5%')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Error Rate (%)')
            ax.set_title('Error Rate Over Time')
            ax.set_xticks(range(len(hours)))
            ax.set_xticklabels([h.split()[1] for h in hours], rotation=45)
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig

# === DEMO ===

print("🚦 Production Monitoring Demo\n")
print("="*70)

monitor = RAGMonitor()

# Simulate queries over time
print("Simulating production traffic...\n")

base_time = datetime.now() - timedelta(hours=5)
for i in range(100):
    # Simulate query execution
    query = f"Query {i+1}"
    latency = np.random.exponential(1.5) + 0.5  # Mean 2s
    retrieval_score = np.random.beta(8, 2)  # Mostly high scores
    token_count = int(np.random.normal(500, 150))
    cost = token_count / 1_000_000 * 2.5  # $2.5 per 1M tokens

    # Occasional errors (5%)
    error = "Timeout" if np.random.random() < 0.05 else None

    # Occasional spikes
    if i % 20 == 0:
        latency *= 3

    monitor.log_query(query, latency, retrieval_score, token_count, cost, error)

    if (i + 1) % 20 == 0:
        print(f"  Logged {i+1} queries...")

# Get statistics
stats = monitor.get_statistics()

print("\n📊 STATISTICS:")
print(f"   Total queries: {stats['total_queries']}")
print(f"   Latency P50: {stats['latency_p50']:.2f}s")
print(f"   Latency P95: {stats['latency_p95']:.2f}s")
print(f"   Latency P99: {stats['latency_p99']:.2f}s")
print(f"   Avg retrieval score: {stats['avg_retrieval_score']:.3f}")
print(f"   Total tokens: {stats['total_tokens']:,}")
print(f"   Total cost: ${stats['total_cost']:.4f}")
print(f"   Error rate: {stats['error_rate']*100:.1f}%")
print(f"   Queries/hour: {stats['queries_per_hour']:.1f}")

# Check alerts
alerts = monitor.get_alerts()

if alerts:
    print(f"\n🚨 ALERTS ({len(alerts)}):")
    for alert in alerts:
        severity_emoji = '🔴' if alert['severity'] == 'critical' else '🟡'
        print(f"   {severity_emoji} [{alert['severity'].upper()}] {alert['message']}")
else:
    print("\n✅ No alerts - all systems nominal")

# Export Prometheus metrics
print("\n📈 Prometheus Metrics:")
print(monitor.export_prometheus_metrics()[:300] + "\n...")

# Visualize dashboard
print("\n📊 Generating monitoring dashboard...")
fig = monitor.visualize_dashboard()
plt.savefig('rag_monitoring_dashboard.png', dpi=150, bbox_inches='tight')
print("✅ Dashboard saved: rag_monitoring_dashboard.png")

print("\n✅ Production monitoring system implemented!")
print("\n💡 INTEGRATION:")
print("""
# Prometheus scrape config
scrape_configs:
  - job_name: 'rag_system'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

# Grafana alerts
- alert: HighLatency
  expr: rag_latency_seconds{quantile="0.95"} > 5
  for: 5m
  annotations:
    summary: "RAG P95 latency is high"

- alert: HighErrorRate
  expr: rag_error_rate > 0.05
  for: 5m
  annotations:
    summary: "RAG error rate > 5%"
""")
