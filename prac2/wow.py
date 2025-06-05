import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform


np.random.seed(42)


n = 100  


company_types = {
    'Tech_Growth': {
        'count': n // 5,  
        'return_params': (0.20, 0.08),  
        'volatility_params': (0.45, 0.08),  
        'marketcap_params': (11.5, 0.4),  
        'pe_params': (45, 8),  
        'dividend_params': (0.003, 0.002)  
    },
    'Traditional_Banks': {
        'count': n // 5,
        'return_params': (0.06, 0.06),  
        'volatility_params': (0.15, 0.04),  
        'marketcap_params': (10.2, 0.3),  
        'pe_params': (8, 3),  
        'dividend_params': (0.055, 0.010)  
    },
    'Oil_Energy': {
        'count': n // 5,
        'return_params': (0.02, 0.25),  
        'volatility_params': (0.55, 0.10),  
        'marketcap_params': (9.8, 0.8),  
        'pe_params': (12, 4),  
        'dividend_params': (0.065, 0.015)  
    },
    'Healthcare': {
        'count': n // 5,
        'return_params': (0.12, 0.05),  
        'volatility_params': (0.12, 0.03),  
        'marketcap_params': (11.0, 0.3),  
        'pe_params': (28, 5),  
        'dividend_params': (0.018, 0.005)  
    },
    'Utilities': {
        'count': n - 4 * (n // 5),  
        'return_params': (0.04, 0.03),  
        'volatility_params': (0.08, 0.02),  
        'marketcap_params': (9.2, 0.25),  
        'pe_params': (15, 3),  
        'dividend_params': (0.08, 0.012)  
    }
}


data = []
company_counter = 1

for group_name, params in company_types.items():
    print(f"Генерируем {params['count']} компаний типа {group_name}")
    
    for i in range(params['count']):
        data.append({
            'Company': f'{group_name}_{i+1}',
            'Group': group_name,
            'Return': np.random.normal(*params['return_params']),
            'Volatility': np.random.normal(*params['volatility_params']),
            'MarketCap': np.random.lognormal(*params['marketcap_params']),
            'PE_Ratio': np.random.normal(*params['pe_params']),
            'Dividend': np.random.normal(*params['dividend_params'])
        })
        company_counter += 1

df = pd.DataFrame(data)


print("Исходные данные:")
print(df.head())
print(f"Пропуски: {df.isnull().sum().sum()}")


df['Volatility'] = df['Volatility'].abs()
df['MarketCap'] = df['MarketCap'].abs() 
df['PE_Ratio'] = df['PE_Ratio'].abs()
df['Dividend'] = df['Dividend'].abs()


features = ['Return', 'Volatility', 'MarketCap', 'PE_Ratio', 'Dividend']
X = df[features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Данные подготовлены: {X.shape}")


print("\n=== АНАЛИЗ РАЗДЕЛИМОСТИ ГРУПП ===")
group_means = df.groupby('Group')[features].mean()
print("Средние значения по группам:")
print(group_means.round(3))


group_distances = pdist(group_means.values, metric='euclidean')
distance_matrix = squareform(group_distances)

print(f"\nМинимальное расстояние между группами: {group_distances.min():.3f}")
print(f"Максимальное расстояние между группами: {group_distances.max():.3f}")
print(f"Среднее расстояние между группами: {group_distances.mean():.3f}")




plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title('Корреляция признаков')


print(X_scaled.shape, ': \tX_scaled.shape')
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.subplot(2, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.7)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.title('PCA компоненты')


inertias = []
K = range(1, 8)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.subplot(2, 3, 3)
plt.plot(K, inertias, 'bo-')
plt.xlabel('Количество кластеров')
plt.ylabel('Инерция')
plt.title('Метод локтя')


silhouette_scores = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)

plt.subplot(2, 3, 4)
plt.plot(range(2, 8), silhouette_scores, 'ro-')
plt.xlabel('Количество кластеров')
plt.ylabel('Silhouette Score')
plt.title('Качество кластеризации')

optimal_k = range(2, 8)[np.argmax(silhouette_scores)]
print(f"Оптимальное количество кластеров: {optimal_k}")




kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)


dbscan = DBSCAN(eps=0.8, min_samples=3)
dbscan_labels = dbscan.fit_predict(X_scaled)


plt.subplot(2, 3, 5)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.8)
plt.title(f'K-means ({optimal_k} кластеров)')
plt.xlabel('PC1')

plt.subplot(2, 3, 6)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.8)
plt.title(f'DBSCAN ({len(set(dbscan_labels))} кластеров)')
plt.xlabel('PC1')

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 5))


plt.subplot(1, 3, 1)
group_colors = {'Tech_Growth': 0, 'Traditional_Banks': 1, 'Oil_Energy': 2, 'Healthcare': 3, 'Utilities': 4}
real_group_labels = [group_colors[group] for group in df['Group']]
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=real_group_labels, cmap='Set1', alpha=0.8)
plt.title('Реальные группы компаний')
plt.xlabel('PC1')
plt.ylabel('PC2')


plt.subplot(1, 3, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.8)
plt.title(f'K-means кластеры')
plt.xlabel('PC1')


plt.subplot(1, 3, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.3)

centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='x', s=200, linewidths=3)
plt.title('K-means с центроидами')
plt.xlabel('PC1')

plt.tight_layout()
plt.show()



df['Cluster'] = kmeans_labels

print("\n=== АНАЛИЗ КЛАСТЕРОВ ===")


for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    print(f"\nКластер {cluster} ({len(cluster_data)} компаний):")
    print(f"Компании: {', '.join(cluster_data['Company'].head(5).tolist())}{'...' if len(cluster_data) > 5 else ''}")
    
    
    means = cluster_data[features].mean()
    print("Характеристики:")
    for feature in features:
        print(f"  {feature}: {means[feature]:.3f}")
    
    
    group_distribution = cluster_data['Group'].value_counts()
    print("Состав по типам компаний:")
    for group, count in group_distribution.items():
        percentage = (count / len(cluster_data)) * 100
        print(f"  {group}: {count} ({percentage:.1f}%)")


print("\n=== СРАВНЕНИЕ С РЕАЛЬНЫМИ ГРУППАМИ ===")


for group in company_types.keys():
    group_data = df[df['Group'] == group]
    cluster_distribution = group_data['Cluster'].value_counts().sort_index()
    print(f"\n{group} ({len(group_data)} компаний):")
    for cluster, count in cluster_distribution.items():
        percentage = (count / len(group_data)) * 100
        print(f"  Кластер {cluster}: {count} компаний ({percentage:.1f}%)")


print("\n=== ИНТЕРПРЕТАЦИЯ ===")
for cluster in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster]
    means = cluster_data[features].mean()
    
    
    if means['Return'] > 0.1 and means['Volatility'] > 0.3:
        cluster_type = "Агрессивные акции роста"
    elif means['Dividend'] > 0.025:
        cluster_type = "Дивидендные акции"
    elif means['Volatility'] < 0.2:
        cluster_type = "Защитные акции"
    else:
        cluster_type = "Сбалансированные акции"
    
    print(f"Кластер {cluster}: {cluster_type}")


plt.figure(figsize=(10, 6))
for cluster in range(optimal_k):
    cluster_means = df[df['Cluster'] == cluster][features].mean()
    plt.plot(features, cluster_means, 'o-', label=f'Кластер {cluster}', linewidth=2)

plt.xlabel('Признаки')
plt.ylabel('Значения (нормализованные)')
plt.title('Профили кластеров')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nSilhouette Score: {silhouette_score(X_scaled, kmeans_labels):.3f}")


from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


real_labels = [group_colors[group] for group in df['Group']]
ari_score = adjusted_rand_score(real_labels, kmeans_labels)
nmi_score = normalized_mutual_info_score(real_labels, kmeans_labels)

print(f"Adjusted Rand Index: {ari_score:.3f}")
print(f"Normalized Mutual Information: {nmi_score:.3f}")
print("Кластеризация завершена!")


df.to_csv('financial_clusters.csv', index=False)
print("Результаты сохранены в 'financial_clusters.csv'")


summary_data = []
for group in company_types.keys():
    group_data = df[df['Group'] == group]
    summary_data.append({
        'Group': group,
        'Count': len(group_data),
        'Avg_Return': group_data['Return'].mean(),
        'Avg_Volatility': group_data['Volatility'].mean(),
        'Avg_MarketCap': group_data['MarketCap'].mean(),
        'Avg_PE_Ratio': group_data['PE_Ratio'].mean(),
        'Avg_Dividend': group_data['Dividend'].mean(),
        'Most_Common_Cluster': group_data['Cluster'].mode().iloc[0] if len(group_data) > 0 else -1
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('group_summary.csv', index=False)
print("Сводка по группам сохранена в 'group_summary.csv'")