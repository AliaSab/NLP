"""
Модуль для оценки качества кластеризации.
Внутренние и внешние метрики.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score
)
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringEvaluator:
    """Класс для оценки качества кластеризации."""
    
    def __init__(self):
        self.results = {}
    
    def compute_internal_metrics(self, vectors: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Вычисление внутренних метрик."""
        # Удаляем шумовые точки для DBSCAN
        non_noise_mask = labels != -1
        if non_noise_mask.sum() < 2:
            logger.warning("Недостаточно точек для вычисления метрик")
            return {
                'silhouette': -1.0,
                'calinski_harabasz': 0.0,
                'davies_bouldin': float('inf')
            }
        
        vectors_clean = vectors[non_noise_mask]
        labels_clean = labels[non_noise_mask]
        
        # Silhouette Score (чем выше, тем лучше, диапазон [-1, 1])
        try:
            silhouette = silhouette_score(vectors_clean, labels_clean)
        except Exception as e:
            logger.warning(f"Ошибка вычисления Silhouette: {e}")
            silhouette = -1.0
        
        # Calinski-Harabasz Index (чем выше, тем лучше)
        try:
            calinski = calinski_harabasz_score(vectors_clean, labels_clean)
        except Exception as e:
            logger.warning(f"Ошибка вычисления Calinski-Harabasz: {e}")
            calinski = 0.0
        
        # Davies-Bouldin Index (чем ниже, тем лучше)
        try:
            davies = davies_bouldin_score(vectors_clean, labels_clean)
        except Exception as e:
            logger.warning(f"Ошибка вычисления Davies-Bouldin: {e}")
            davies = float('inf')
        
        metrics = {
            'silhouette': float(silhouette),
            'calinski_harabasz': float(calinski),
            'davies_bouldin': float(davies),
            'n_clusters': len(set(labels_clean)),
            'n_noise': int((labels == -1).sum()) if -1 in labels else 0
        }
        
        return metrics
    
    def compute_external_metrics(self, labels_true: np.ndarray, 
                                 labels_pred: np.ndarray) -> Dict[str, float]:
        """Вычисление внешних метрик."""
        # Удаляем шумовые точки
        non_noise_mask = labels_pred != -1
        if non_noise_mask.sum() < 2:
            return {
                'ari': -1.0,
                'nmi': 0.0,
                'v_measure': 0.0
            }
        
        labels_true_clean = labels_true[non_noise_mask]
        labels_pred_clean = labels_pred[non_noise_mask]
        
        # Adjusted Rand Index (ARI)
        try:
            ari = adjusted_rand_score(labels_true_clean, labels_pred_clean)
        except Exception as e:
            logger.warning(f"Ошибка вычисления ARI: {e}")
            ari = -1.0
        
        # Normalized Mutual Information (NMI)
        try:
            nmi = normalized_mutual_info_score(labels_true_clean, labels_pred_clean)
        except Exception as e:
            logger.warning(f"Ошибка вычисления NMI: {e}")
            nmi = 0.0
        
        # V-measure
        try:
            v_measure = v_measure_score(labels_true_clean, labels_pred_clean)
        except Exception as e:
            logger.warning(f"Ошибка вычисления V-measure: {e}")
            v_measure = 0.0
        
        return {
            'ari': float(ari),
            'nmi': float(nmi),
            'v_measure': float(v_measure)
        }
    
    def evaluate_clustering(self, vectors: np.ndarray, labels: np.ndarray,
                           labels_true: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Полная оценка кластеризации."""
        results = {}
        
        # Внутренние метрики
        internal = self.compute_internal_metrics(vectors, labels)
        results['internal'] = internal
        
        # Внешние метрики (если есть истинные метки)
        if labels_true is not None:
            external = self.compute_external_metrics(labels_true, labels)
            results['external'] = external
        
        return results
    
    def plot_metrics_vs_parameters(self, results: Dict[str, Dict[str, Any]],
                                   metric_name: str = 'silhouette',
                                   param_name: str = 'n_clusters',
                                   save_path: Optional[str] = None):
        """Построение графика зависимости метрики от параметров."""
        param_values = []
        metric_values = []
        
        for key, metrics in results.items():
            if param_name in metrics and metric_name in metrics:
                param_values.append(metrics[param_name])
                metric_values.append(metrics[metric_name])
        
        if not param_values:
            logger.warning("Нет данных для построения графика")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(param_values, metric_values, marker='o', linewidth=2, markersize=8)
        plt.xlabel(param_name, fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f'Зависимость {metric_name} от {param_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"График сохранен: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_methods(self, vectors: np.ndarray, 
                       labels_dict: Dict[str, np.ndarray],
                       labels_true: Optional[np.ndarray] = None) -> Dict[str, Dict[str, Any]]:
        """Сравнение различных методов кластеризации."""
        comparison = {}
        
        for method_name, labels in labels_dict.items():
            logger.info(f"Оценка метода: {method_name}")
            results = self.evaluate_clustering(vectors, labels, labels_true)
            comparison[method_name] = results
        
        return comparison
    
    def print_metrics(self, results: Dict[str, Any]):
        """Вывод метрик в консоль."""
        print("\n" + "="*50)
        print("МЕТРИКИ КАЧЕСТВА КЛАСТЕРИЗАЦИИ")
        print("="*50)
        
        if 'internal' in results:
            print("\nВнутренние метрики:")
            internal = results['internal']
            print(f"  Silhouette Score: {internal['silhouette']:.4f} (чем выше, тем лучше)")
            print(f"  Calinski-Harabasz Index: {internal['calinski_harabasz']:.4f} (чем выше, тем лучше)")
            print(f"  Davies-Bouldin Index: {internal['davies_bouldin']:.4f} (чем ниже, тем лучше)")
            print(f"  Количество кластеров: {internal['n_clusters']}")
            if internal['n_noise'] > 0:
                print(f"  Шумовых точек: {internal['n_noise']}")
        
        if 'external' in results:
            print("\nВнешние метрики:")
            external = results['external']
            print(f"  Adjusted Rand Index (ARI): {external['ari']:.4f} (чем выше, тем лучше)")
            print(f"  Normalized Mutual Information (NMI): {external['nmi']:.4f} (чем выше, тем лучше)")
            print(f"  V-measure: {external['v_measure']:.4f} (чем выше, тем лучше)")
        
        print("="*50 + "\n")


