"""
Модуль для кластеризации текстов.
Поддерживает различные алгоритмы кластеризации.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    try:
        import umap.umap_ as umap_module
        UMAP = umap_module.UMAP
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
        UMAP = None
        logging.warning("UMAP не установлен. Установите: pip install umap-learn")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringPipeline:
    """Класс для кластеризации текстов."""
    
    def __init__(self):
        self.models = {}
        self.labels = {}
        self.centroids = {}
    
    def kmeans(self, vectors: np.ndarray, n_clusters: int = 5, 
               random_state: int = 42, n_init: int = 10) -> np.ndarray:
        """K-Means кластеризация."""
        logger.info(f"K-Means кластеризация: n_clusters={n_clusters}")
        
        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=n_init,
            max_iter=300
        )
        
        labels = model.fit_predict(vectors)
        self.models['kmeans'] = model
        self.labels['kmeans'] = labels
        self.centroids['kmeans'] = model.cluster_centers_
        
        return labels
    
    def dbscan(self, vectors: np.ndarray, eps: float = 0.5, 
               min_samples: int = 5) -> np.ndarray:
        """DBSCAN кластеризация."""
        logger.info(f"DBSCAN кластеризация: eps={eps}, min_samples={min_samples}")
        
        model = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = model.fit_predict(vectors)
        
        self.models['dbscan'] = model
        self.labels['dbscan'] = labels
        
        # Вычисляем центроиды для не-шумовых кластеров
        unique_labels = set(labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        centroids = []
        for label in sorted(unique_labels):
            cluster_points = vectors[labels == label]
            centroids.append(np.mean(cluster_points, axis=0))
        
        if centroids:
            self.centroids['dbscan'] = np.array(centroids)
        
        return labels
    
    def hierarchical(self, vectors: np.ndarray, n_clusters: int = 5,
                    linkage: str = 'ward', affinity: str = 'euclidean') -> np.ndarray:
        """Иерархическая кластеризация."""
        logger.info(f"Иерархическая кластеризация: n_clusters={n_clusters}, linkage={linkage}")
        
        # Для 'ward' linkage можно использовать только 'euclidean'
        if linkage == 'ward':
            affinity = 'euclidean'
        
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
            affinity=affinity
        )
        
        labels = model.fit_predict(vectors)
        self.models['hierarchical'] = model
        self.labels['hierarchical'] = labels
        
        # Вычисляем центроиды
        centroids = []
        for label in range(n_clusters):
            cluster_points = vectors[labels == label]
            centroids.append(np.mean(cluster_points, axis=0))
        self.centroids['hierarchical'] = np.array(centroids)
        
        return labels
    
    def gmm(self, vectors: np.ndarray, n_components: int = 5,
           random_state: int = 42, covariance_type: str = 'full') -> np.ndarray:
        """Gaussian Mixture Model кластеризация."""
        logger.info(f"GMM кластеризация: n_components={n_components}")
        
        model = GaussianMixture(
            n_components=n_components,
            random_state=random_state,
            covariance_type=covariance_type,
            max_iter=100
        )
        
        labels = model.predict(vectors)
        self.models['gmm'] = model
        self.labels['gmm'] = labels
        self.centroids['gmm'] = model.means_
        
        return labels
    
    def spectral(self, vectors: np.ndarray, n_clusters: int = 5,
                random_state: int = 42, n_neighbors: int = 10) -> np.ndarray:
        """Spectral кластеризация."""
        logger.info(f"Spectral кластеризация: n_clusters={n_clusters}")
        
        model = SpectralClustering(
            n_clusters=n_clusters,
            random_state=random_state,
            affinity='nearest_neighbors',
            n_neighbors=n_neighbors
        )
        
        labels = model.fit_predict(vectors)
        self.models['spectral'] = model
        self.labels['spectral'] = labels
        
        # Вычисляем центроиды
        centroids = []
        for label in range(n_clusters):
            cluster_points = vectors[labels == label]
            centroids.append(np.mean(cluster_points, axis=0))
        self.centroids['spectral'] = np.array(centroids)
        
        return labels
    
    def reduce_dimensions(self, vectors: np.ndarray, method: str = 'pca',
                         n_components: int = 2, random_state: int = 42) -> np.ndarray:
        """Снижение размерности для визуализации."""
        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=random_state)
            reduced = reducer.fit_transform(vectors)
            logger.info(f"PCA: объясненная дисперсия = {reducer.explained_variance_ratio_.sum():.4f}")
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=random_state, perplexity=30)
            reduced = reducer.fit_transform(vectors)
        elif method == 'umap':
            if not UMAP_AVAILABLE or UMAP is None:
                raise ImportError("UMAP не установлен. Установите: pip install umap-learn")
            reducer = UMAP(n_components=n_components, random_state=random_state)
            reduced = reducer.fit_transform(vectors)
        else:
            raise ValueError(f"Неизвестный метод снижения размерности: {method}")
        
        return reduced
    
    def get_cluster_centroids(self, method: str) -> Optional[np.ndarray]:
        """Получение центроидов кластеров."""
        return self.centroids.get(method)
    
    def get_labels(self, method: str) -> Optional[np.ndarray]:
        """Получение меток кластеров."""
        return self.labels.get(method)


def tune_kmeans(vectors: np.ndarray, k_range: List[int] = range(2, 11),
                random_state: int = 42) -> Dict[int, Dict[str, float]]:
    """Подбор оптимального числа кластеров для K-Means."""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    results = {}
    
    for k in k_range:
        logger.info(f"Тестирование K-Means с k={k}")
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(vectors)
        
        # Вычисляем метрики
        silhouette = silhouette_score(vectors, labels)
        calinski = calinski_harabasz_score(vectors, labels)
        davies = davies_bouldin_score(vectors, labels)
        
        results[k] = {
            'silhouette': silhouette,
            'calinski_harabasz': calinski,
            'davies_bouldin': davies,
            'inertia': kmeans.inertia_
        }
    
    return results


def tune_dbscan(vectors: np.ndarray, eps_range: List[float] = None,
                min_samples_range: List[int] = None) -> Dict[str, Dict[str, Any]]:
    """Подбор оптимальных параметров для DBSCAN."""
    from sklearn.metrics import silhouette_score
    
    if eps_range is None:
        eps_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if min_samples_range is None:
        min_samples_range = [3, 5, 7, 10]
    
    results = {}
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            logger.info(f"Тестирование DBSCAN: eps={eps}, min_samples={min_samples}")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
            labels = dbscan.fit_predict(vectors)
            
            # Пропускаем если все точки - шум
            unique_labels = set(labels)
            if len(unique_labels) < 2 or (len(unique_labels) == 2 and -1 in unique_labels):
                continue
            
            # Вычисляем метрики только для не-шумовых точек
            non_noise_mask = labels != -1
            if non_noise_mask.sum() < 2:
                continue
            
            silhouette = silhouette_score(vectors[non_noise_mask], labels[non_noise_mask])
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            n_noise = (labels == -1).sum()
            
            key = f"eps_{eps}_min_{min_samples}"
            results[key] = {
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'silhouette': silhouette
            }
    
    return results


def tune_hierarchical(vectors: np.ndarray, n_clusters_range: List[int] = None,
                      linkage_options: List[str] = None, random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    """Подбор оптимальных параметров для иерархической кластеризации."""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    if n_clusters_range is None:
        n_clusters_range = list(range(2, min(11, len(vectors) // 10 + 1)))
    if linkage_options is None:
        linkage_options = ['ward', 'complete', 'average']
    
    results = {}
    
    for n_clusters in n_clusters_range:
        for linkage in linkage_options:
            try:
                logger.info(f"Тестирование Hierarchical: n_clusters={n_clusters}, linkage={linkage}")
                
                # Для 'ward' linkage можно использовать только 'euclidean'
                affinity = 'euclidean' if linkage == 'ward' else 'cosine'
                
                model = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage=linkage,
                    affinity=affinity
                )
                labels = model.fit_predict(vectors)
                
                # Вычисляем метрики
                silhouette = silhouette_score(vectors, labels)
                calinski = calinski_harabasz_score(vectors, labels)
                davies = davies_bouldin_score(vectors, labels)
                
                key = f"n_{n_clusters}_link_{linkage}"
                results[key] = {
                    'n_clusters': n_clusters,
                    'linkage': linkage,
                    'silhouette': silhouette,
                    'calinski_harabasz': calinski,
                    'davies_bouldin': davies
                }
            except Exception as e:
                logger.warning(f"Ошибка при тестировании Hierarchical: {e}")
                continue
    
    return results


def tune_gmm(vectors: np.ndarray, n_components_range: List[int] = None,
             random_state: int = 42) -> Dict[int, Dict[str, float]]:
    """Подбор оптимального числа компонент для GMM."""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    if n_components_range is None:
        n_components_range = list(range(2, min(11, len(vectors) // 10 + 1)))
    
    results = {}
    
    for n_components in n_components_range:
        try:
            logger.info(f"Тестирование GMM с n_components={n_components}")
            gmm = GaussianMixture(
                n_components=n_components,
                random_state=random_state,
                covariance_type='full',
                max_iter=100
            )
            labels = gmm.predict(vectors)
            
            # Вычисляем метрики
            silhouette = silhouette_score(vectors, labels)
            calinski = calinski_harabasz_score(vectors, labels)
            davies = davies_bouldin_score(vectors, labels)
            
            results[n_components] = {
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies,
                'aic': gmm.aic(vectors),
                'bic': gmm.bic(vectors)
            }
        except Exception as e:
            logger.warning(f"Ошибка при тестировании GMM: {e}")
            continue
    
    return results


def tune_spectral(vectors: np.ndarray, n_clusters_range: List[int] = None,
                  random_state: int = 42) -> Dict[int, Dict[str, float]]:
    """Подбор оптимального числа кластеров для Spectral кластеризации."""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    
    if n_clusters_range is None:
        n_clusters_range = list(range(2, min(11, len(vectors) // 10 + 1)))
    
    results = {}
    
    for n_clusters in n_clusters_range:
        try:
            logger.info(f"Тестирование Spectral с n_clusters={n_clusters}")
            spectral = SpectralClustering(
                n_clusters=n_clusters,
                random_state=random_state,
                affinity='nearest_neighbors',
                n_neighbors=10
            )
            labels = spectral.fit_predict(vectors)
            
            # Вычисляем метрики
            silhouette = silhouette_score(vectors, labels)
            calinski = calinski_harabasz_score(vectors, labels)
            davies = davies_bouldin_score(vectors, labels)
            
            results[n_clusters] = {
                'silhouette': silhouette,
                'calinski_harabasz': calinski,
                'davies_bouldin': davies
            }
        except Exception as e:
            logger.warning(f"Ошибка при тестировании Spectral: {e}")
            continue
    
    return results

