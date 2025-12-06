"""
Модуль анализа и снижения размерности векторных пространств.

Реализует:
- Применение SVD (LSA) для выявления скрытых семантических тем
- Снижение размерности с контролем сохраняемой дисперсии
- Визуализацию компонент и их интерпретацию
- Анализ зависимости качества представлений от числа компонент
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
# Попытка импорта UMAP с обработкой различных версий
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    try:
        import umap.umap_ as umap
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
        print("UMAP не установлен. Визуализация UMAP будет пропущена.")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import csr_matrix
import json
import time
from typing import List, Dict, Tuple, Any, Optional
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class DimensionalityReducer:
    """Класс для снижения размерности и тематического моделирования."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        
    def svd_analysis(self, matrix: csr_matrix, n_components_range: List[int] = None) -> Dict[str, Any]:
        """Анализ SVD (LSA) с различным количеством компонент."""
        if n_components_range is None:
            n_components_range = [10, 50, 100, 200, 500]
        
        logger.info(f"Анализ SVD с компонентами: {n_components_range}")
        
        results = {}
        
        for n_components in n_components_range:
            if n_components >= min(matrix.shape):
                continue
                
            logger.info(f"Обработка SVD с {n_components} компонентами")
            
            start_time = time.time()
            
            # Применение SVD
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            transformed_matrix = svd.fit_transform(matrix)
            
            processing_time = time.time() - start_time
            
            # Анализ объясненной дисперсии
            explained_variance_ratio = svd.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Анализ сингулярных значений
            singular_values = svd.singular_values_
            
            results[n_components] = {
                'model': svd,
                'transformed_matrix': transformed_matrix,
                'explained_variance_ratio': explained_variance_ratio,
                'cumulative_variance': cumulative_variance,
                'singular_values': singular_values,
                'processing_time': processing_time,
                'n_components': n_components
            }
            
            logger.info(f"SVD с {n_components} компонентами: объясненная дисперсия {cumulative_variance[-1]:.3f}")
        
        return results
    
    def analyze_variance_explained(self, svd_results: Dict[int, Any]) -> pd.DataFrame:
        """Анализ объясненной дисперсии для различных количеств компонент."""
        variance_data = []
        
        for n_components, result in svd_results.items():
            cumulative_variance = result['cumulative_variance']
            explained_variance = result['explained_variance_ratio']
            
            variance_data.append({
                'n_components': n_components,
                'total_variance_explained': cumulative_variance[-1],
                'variance_90_percent': np.where(cumulative_variance >= 0.9)[0][0] + 1 if len(np.where(cumulative_variance >= 0.9)[0]) > 0 else n_components,
                'variance_95_percent': np.where(cumulative_variance >= 0.95)[0][0] + 1 if len(np.where(cumulative_variance >= 0.95)[0]) > 0 else n_components,
                'variance_99_percent': np.where(cumulative_variance >= 0.99)[0][0] + 1 if len(np.where(cumulative_variance >= 0.99)[0]) > 0 else n_components,
                'first_component_variance': explained_variance[0],
                'top_5_components_variance': np.sum(explained_variance[:5]),
                'processing_time': result.get('processing_time', 0.0)
            })
        
        return pd.DataFrame(variance_data)
    
    def visualize_components(self, svd_results: Dict[int, Any], feature_names: List[str], 
                           top_n: int = 20) -> None:
        """Визуализация компонент SVD."""
        logger.info("Создание визуализации компонент")
        
        # Создание графиков для каждого количества компонент
        for n_components, result in svd_results.items():
            if n_components > 100:  # Ограничиваем для больших размерностей
                continue
                
            model = result['model']
            components = model.components_
            
            # Создание subplot для первых нескольких компонент
            n_plots = min(6, n_components)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'SVD Компоненты (n_components={n_components})', fontsize=16)
            
            for i in range(n_plots):
                row = i // 3
                col = i % 3
                ax = axes[row, col]
                
                # Получение топ признаков для компоненты
                component_weights = components[i]
                top_indices = np.argsort(np.abs(component_weights))[-top_n:]
                top_weights = component_weights[top_indices]
                top_features = [feature_names[idx] for idx in top_indices]
                
                # Создание горизонтального bar plot
                y_pos = np.arange(len(top_features))
                ax.barh(y_pos, top_weights)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(top_features, fontsize=8)
                ax.set_xlabel('Вес компоненты')
                ax.set_title(f'Компонента {i+1}')
                ax.grid(True, alpha=0.3)
            
            # Скрытие пустых subplot
            for i in range(n_plots, 6):
                row = i // 3
                col = i % 3
                axes[row, col].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'svd_components_{n_components}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def visualize_variance_explained(self, variance_df: pd.DataFrame) -> None:
        """Визуализация объясненной дисперсии."""
        logger.info("Создание визуализации объясненной дисперсии")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Анализ объясненной дисперсии SVD', fontsize=16)
        
        # График общей объясненной дисперсии
        axes[0, 0].plot(variance_df['n_components'], variance_df['total_variance_explained'], 'bo-')
        axes[0, 0].set_xlabel('Количество компонент')
        axes[0, 0].set_ylabel('Общая объясненная дисперсия')
        axes[0, 0].set_title('Общая объясненная дисперсия')
        axes[0, 0].grid(True, alpha=0.3)
        
        # График компонент для достижения 90%, 95%, 99% дисперсии
        axes[0, 1].plot(variance_df['n_components'], variance_df['variance_90_percent'], 'ro-', label='90%')
        axes[0, 1].plot(variance_df['n_components'], variance_df['variance_95_percent'], 'go-', label='95%')
        axes[0, 1].plot(variance_df['n_components'], variance_df['variance_99_percent'], 'bo-', label='99%')
        axes[0, 1].set_xlabel('Количество компонент')
        axes[0, 1].set_ylabel('Компоненты для достижения порога')
        axes[0, 1].set_title('Компоненты для достижения порогов дисперсии')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # График дисперсии первой компоненты
        axes[1, 0].plot(variance_df['n_components'], variance_df['first_component_variance'], 'mo-')
        axes[1, 0].set_xlabel('Количество компонент')
        axes[1, 0].set_ylabel('Дисперсия первой компоненты')
        axes[1, 0].set_title('Дисперсия первой компоненты')
        axes[1, 0].grid(True, alpha=0.3)
        
        # График времени обработки
        axes[1, 1].plot(variance_df['n_components'], variance_df['processing_time'], 'co-')
        axes[1, 1].set_xlabel('Количество компонент')
        axes[1, 1].set_ylabel('Время обработки (сек)')
        axes[1, 1].set_title('Время обработки')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('svd_variance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def clustering_analysis(self, transformed_matrices: Dict[int, np.ndarray], 
                           n_clusters_range: List[int] = None) -> Dict[str, Any]:
        """Анализ качества кластеризации для различных размерностей."""
        if n_clusters_range is None:
            n_clusters_range = [5, 10, 15, 20, 25]
        
        logger.info("Анализ качества кластеризации")
        
        clustering_results = {}
        
        for n_components, matrix in transformed_matrices.items():
            clustering_results[n_components] = {}
            
            for n_clusters in n_clusters_range:
                if n_clusters >= matrix.shape[0]:
                    continue
                    
                logger.info(f"Кластеризация: {n_components} компонент, {n_clusters} кластеров")
                
                # K-means кластеризация
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(matrix)
                
                # Оценка качества кластеризации
                silhouette_avg = silhouette_score(matrix, cluster_labels)
                inertia = kmeans.inertia_
                
                clustering_results[n_components][n_clusters] = {
                    'silhouette_score': silhouette_avg,
                    'inertia': inertia,
                    'cluster_labels': cluster_labels,
                    'cluster_centers': kmeans.cluster_centers_
                }
        
        return clustering_results
    
    def analyze_representation_quality(self, svd_results: Dict[int, Any], 
                                     original_matrix: csr_matrix) -> Dict[str, Any]:
        """Анализ зависимости качества представлений от числа компонент."""
        logger.info("Анализ качества представлений")
        
        quality_results = {}
        
        for n_components, result in svd_results.items():
            try:
                # Восстановление матрицы
                reconstructed = result['model'].inverse_transform(result['transformed_matrix'])
                
                # Вычисление ошибки восстановления
                reconstruction_error = np.mean((original_matrix.toarray() - reconstructed) ** 2)
                
                # Вычисление объясненной дисперсии
                explained_variance = np.sum(result['explained_variance_ratio'])
                
                # Вычисление эффективности сжатия
                compression_ratio = original_matrix.shape[1] / n_components
                
                quality_results[n_components] = {
                    'reconstruction_error': reconstruction_error,
                    'explained_variance': explained_variance,
                    'compression_ratio': compression_ratio,
                    'processing_time': result.get('processing_time', 0.0)
                }
                
            except Exception as e:
                logger.warning(f"Ошибка анализа качества для {n_components} компонент: {e}")
                continue
        
        return quality_results
    
    def visualize_clustering_quality(self, clustering_results: Dict[str, Any]) -> None:
        """Визуализация качества кластеризации."""
        logger.info("Создание визуализации качества кластеризации")
        
        # Подготовка данных для heatmap
        silhouette_scores = []
        inertias = []
        n_components_list = []
        n_clusters_list = []
        
        for n_components, clusters_data in clustering_results.items():
            for n_clusters, metrics in clusters_data.items():
                silhouette_scores.append(metrics['silhouette_score'])
                inertias.append(metrics['inertia'])
                n_components_list.append(n_components)
                n_clusters_list.append(n_clusters)
        
        # Создание DataFrame для heatmap
        df_silhouette = pd.DataFrame({
            'n_components': n_components_list,
            'n_clusters': n_clusters_list,
            'silhouette_score': silhouette_scores
        }).pivot(index='n_components', columns='n_clusters', values='silhouette_score')
        
        df_inertia = pd.DataFrame({
            'n_components': n_components_list,
            'n_clusters': n_clusters_list,
            'inertia': inertias
        }).pivot(index='n_components', columns='n_clusters', values='inertia')
        
        # Создание графиков
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Анализ качества кластеризации', fontsize=16)
        
        # Heatmap silhouette scores
        sns.heatmap(df_silhouette, annot=True, fmt='.3f', cmap='viridis', ax=axes[0])
        axes[0].set_title('Silhouette Score')
        axes[0].set_xlabel('Количество кластеров')
        axes[0].set_ylabel('Количество компонент SVD')
        
        # Heatmap inertias
        sns.heatmap(df_inertia, annot=True, fmt='.0f', cmap='plasma', ax=axes[1])
        axes[1].set_title('Inertia (Within-cluster Sum of Squares)')
        axes[1].set_xlabel('Количество кластеров')
        axes[1].set_ylabel('Количество компонент SVD')
        
        plt.tight_layout()
        plt.savefig('clustering_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def dimensionality_reduction_visualization(self, matrix: csr_matrix, 
                                            method: str = 'tsne') -> None:
        """Визуализация снижения размерности с помощью t-SNE или UMAP."""
        logger.info(f"Создание визуализации снижения размерности методом {method}")
        
        # Предварительное снижение размерности с помощью SVD
        svd = TruncatedSVD(n_components=50, random_state=42)
        matrix_reduced = svd.fit_transform(matrix)
        
        if method == 'tsne':
            # t-SNE
            try:
                perplexity = min(30, len(matrix_reduced) - 1)
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                matrix_2d = tsne.fit_transform(matrix_reduced)
            except Exception as e:
                logger.warning(f"Ошибка с t-SNE: {e}, используем PCA")
                pca = PCA(n_components=2, random_state=42)
                matrix_2d = pca.fit_transform(matrix_reduced)
                method = 'pca_fallback'
        elif method == 'umap':
            if not UMAP_AVAILABLE:
                logger.warning("UMAP не доступен, пропускаем визуализацию UMAP")
                return
            
            # UMAP
            try:
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(matrix_reduced)-1))
                matrix_2d = reducer.fit_transform(matrix_reduced)
            except Exception as e:
                logger.warning(f"Ошибка с UMAP: {e}, используем PCA вместо UMAP")
                # Fallback к PCA
                pca = PCA(n_components=2, random_state=42)
                matrix_2d = pca.fit_transform(matrix_reduced)
                method = 'pca_fallback'
        else:
            logger.warning(f"Неизвестный метод {method}, используем PCA")
            pca = PCA(n_components=2, random_state=42)
            matrix_2d = pca.fit_transform(matrix_reduced)
            method = 'pca'
        
        # Создание scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(matrix_2d[:, 0], matrix_2d[:, 1], alpha=0.6, s=20)
        plt.title(f'2D Визуализация документов ({method.upper()})')
        plt.xlabel('Компонента 1')
        plt.ylabel('Компонента 2')
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f'dimensionality_reduction_{method}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_comprehensive_analysis(self, matrix: csr_matrix, feature_names: List[str]) -> Dict[str, Any]:
        """Запуск комплексного анализа снижения размерности."""
        logger.info("Запуск комплексного анализа снижения размерности")
        
        results = {}
        
        # SVD анализ
        logger.info("Выполнение SVD анализа")
        svd_results = self.svd_analysis(matrix)
        results['svd'] = svd_results
        
        # Анализ объясненной дисперсии
        variance_df = None
        try:
            variance_df = self.analyze_variance_explained(svd_results)
            results['variance_analysis'] = variance_df
        except Exception as e:
            logger.error(f"Ошибка анализа дисперсии: {e}")
            results['variance_analysis'] = pd.DataFrame()
            variance_df = pd.DataFrame()
        
        # Визуализация компонент
        try:
            self.visualize_components(svd_results, feature_names)
        except Exception as e:
            logger.error(f"Ошибка визуализации компонент: {e}")
        
        # Визуализация объясненной дисперсии
        try:
            if variance_df is not None and not variance_df.empty:
                self.visualize_variance_explained(variance_df)
        except Exception as e:
            logger.error(f"Ошибка визуализации дисперсии: {e}")
        
        # Анализ качества представлений
        logger.info("Анализ качества представлений")
        try:
            quality_results = self.analyze_representation_quality(svd_results, matrix)
            results['quality_analysis'] = quality_results
        except Exception as e:
            logger.error(f"Ошибка анализа качества: {e}")
            results['quality_analysis'] = {}
        
        # Анализ кластеризации
        logger.info("Выполнение анализа кластеризации")
        try:
            transformed_matrices = {n_comp: result['transformed_matrix'] 
                                  for n_comp, result in svd_results.items()}
            clustering_results = self.clustering_analysis(transformed_matrices)
            results['clustering'] = clustering_results
        except Exception as e:
            logger.error(f"Ошибка анализа кластеризации: {e}")
            results['clustering'] = {}
        
        # Визуализация качества кластеризации
        if results['clustering']:
            self.visualize_clustering_quality(results['clustering'])
        
        # Визуализация снижения размерности
        try:
            self.dimensionality_reduction_visualization(matrix, 'tsne')
        except Exception as e:
            logger.error(f"Ошибка t-SNE визуализации: {e}")
        
        if UMAP_AVAILABLE:
            try:
                self.dimensionality_reduction_visualization(matrix, 'umap')
            except Exception as e:
                logger.error(f"Ошибка UMAP визуализации: {e}")
        else:
            logger.info("UMAP не доступен, пропускаем UMAP визуализацию")
        
        # Сохранение результатов
        self.save_analysis_results(results)
        
        return results
    
    def save_analysis_results(self, results: Dict[str, Any]) -> None:
        """Сохранение результатов анализа."""
        logger.info("Сохранение результатов анализа")
        
        # Подготовка данных для сохранения
        save_data = {
            'variance_analysis': results.get('variance_analysis', pd.DataFrame()).to_dict('records'),
            'quality_analysis': results.get('quality_analysis', {}),
            'clustering_summary': {}
        }
        
        # Сводка по кластеризации
        clustering_results = results.get('clustering', {})
        for n_components, clusters_data in clustering_results.items():
            save_data['clustering_summary'][str(n_components)] = {}
            for n_clusters, metrics in clusters_data.items():
                save_data['clustering_summary'][str(n_components)][str(n_clusters)] = {
                    'silhouette_score': float(metrics.get('silhouette_score', 0.0)),
                    'inertia': float(metrics.get('inertia', 0.0))
                }
        
        with open('dimensionality_reduction_results.json', 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        # Сохранение таблицы анализа дисперсии
        variance_analysis = results.get('variance_analysis')
        if variance_analysis is not None and not variance_analysis.empty:
            variance_analysis.to_csv('variance_analysis.csv', index=False, encoding='utf-8')
        
        logger.info("Результаты сохранены в:")
        logger.info("- dimensionality_reduction_results.json")
        logger.info("- variance_analysis.csv")


def main():
    """Основная функция для демонстрации работы модуля."""
    from classical_vectorizers import ClassicalVectorizer
    
    # Загрузка данных и создание матрицы
    vectorizer = ClassicalVectorizer()
    texts = vectorizer.load_data("../Task1/kommersant_articles_processed.jsonl")
    
    if not texts:
        logger.error("Не удалось загрузить данные")
        return
    
    # Ограничиваем количество текстов
    texts = texts[:1000]
    
    # Создание TF-IDF матрицы
    logger.info("Создание TF-IDF матрицы")
    tfidf_result = vectorizer.tfidf_vectorization(texts, ngram_range=(1, 2))
    matrix = tfidf_result['matrix']
    feature_names = tfidf_result['feature_names']
    
    # Запуск анализа снижения размерности
    reducer = DimensionalityReducer()
    results = reducer.run_comprehensive_analysis(matrix, feature_names)
    
    # Вывод сводной статистики
    print("\n=== АНАЛИЗ СНИЖЕНИЯ РАЗМЕРНОСТИ ===")
    print(results['variance_analysis'].to_string(index=False))


if __name__ == "__main__":
    main()
