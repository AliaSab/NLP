"""
Модуль для интерпретации результатов кластеризации.
Топ слова, ближайшие слова к центроидам, визуализация.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import Counter
import matplotlib.pyplot as plt
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
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringInterpreter:
    """Класс для интерпретации результатов кластеризации."""
    
    def __init__(self, vectorizer=None):
        self.vectorizer = vectorizer
        self.tokenized_texts = None
        self.vectors = None
        self.labels = None
    
    def set_data(self, tokenized_texts: List[List[str]], 
                 vectors: np.ndarray,
                 labels: np.ndarray):
        """Установка данных для интерпретации."""
        self.tokenized_texts = tokenized_texts
        self.vectors = vectors
        self.labels = labels
    
    def get_top_words_per_cluster_tfidf(self, tfidf_vectorizer, 
                                       n_words: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """Получение топ-N слов для каждого кластера на основе TF-IDF."""
        if self.labels is None or self.tokenized_texts is None:
            raise ValueError("Данные не установлены. Вызовите set_data() сначала.")
        
        top_words = {}
        unique_labels = set(self.labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)  # Исключаем шум
        
        for cluster_id in unique_labels:
            # Индексы документов в кластере
            cluster_indices = np.where(self.labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Получаем TF-IDF векторы для документов кластера
            cluster_texts = [' '.join(self.tokenized_texts[i]) for i in cluster_indices]
            cluster_vectors = tfidf_vectorizer.transform(cluster_texts)
            
            # Суммируем TF-IDF значения по всем документам кластера
            cluster_tfidf_sum = np.array(cluster_vectors.sum(axis=0)).flatten()
            
            # Получаем индексы топ-N слов
            top_indices = cluster_tfidf_sum.argsort()[-n_words:][::-1]
            
            # Получаем слова и их значения
            feature_names = tfidf_vectorizer.get_feature_names_out()
            cluster_top_words = [
                (feature_names[idx], float(cluster_tfidf_sum[idx]))
                for idx in top_indices
            ]
            
            top_words[cluster_id] = cluster_top_words
        
        return top_words
    
    def get_top_words_per_cluster_frequency(self, n_words: int = 10) -> Dict[int, List[Tuple[str, int]]]:
        """Получение топ-N слов для каждого кластера на основе частоты."""
        if self.labels is None or self.tokenized_texts is None:
            raise ValueError("Данные не установлены. Вызовите set_data() сначала.")
        
        top_words = {}
        unique_labels = set(self.labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        
        for cluster_id in unique_labels:
            cluster_indices = np.where(self.labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Собираем все слова из кластера
            all_words = []
            for idx in cluster_indices:
                all_words.extend(self.tokenized_texts[idx])
            
            # Подсчитываем частоту
            word_counts = Counter(all_words)
            top_words[cluster_id] = word_counts.most_common(n_words)
        
        return top_words
    
    def get_similar_words_to_centroid(self, centroids: np.ndarray,
                                     model_type: str = 'word2vec',
                                     top_n: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """Получение ближайших слов к центроидам кластеров через эмбеддинги."""
        if self.vectorizer is None:
            raise ValueError("Векторизатор не установлен")
        
        similar_words = {}
        
        for cluster_id, centroid in enumerate(centroids):
            # Для эмбеддингов нужно найти ближайшие слова
            # Это упрощенная версия - в реальности нужно вычислять расстояния
            # ко всем словам в словаре модели
            
            if model_type in ['word2vec', 'fasttext']:
                # Используем метод векторизатора для поиска похожих слов
                # Это требует доступа к модели напрямую
                try:
                    if model_type == 'word2vec' and self.vectorizer.word2vec_model:
                        # Находим ближайшие слова через косинусное расстояние
                        words = list(self.vectorizer.word2vec_model.wv.key_to_index.keys())
                        similarities = []
                        
                        for word in words[:1000]:  # Ограничиваем для скорости
                            word_vec = self.vectorizer.word2vec_model.wv[word]
                            similarity = np.dot(centroid, word_vec) / (
                                np.linalg.norm(centroid) * np.linalg.norm(word_vec)
                            )
                            similarities.append((word, float(similarity)))
                        
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        similar_words[cluster_id] = similarities[:top_n]
                    elif model_type == 'fasttext' and self.vectorizer.fasttext_model:
                        words = list(self.vectorizer.fasttext_model.wv.key_to_index.keys())
                        similarities = []
                        
                        for word in words[:1000]:
                            word_vec = self.vectorizer.fasttext_model.wv[word]
                            similarity = np.dot(centroid, word_vec) / (
                                np.linalg.norm(centroid) * np.linalg.norm(word_vec)
                            )
                            similarities.append((word, float(similarity)))
                        
                        similarities.sort(key=lambda x: x[1], reverse=True)
                        similar_words[cluster_id] = similarities[:top_n]
                except Exception as e:
                    logger.warning(f"Ошибка поиска похожих слов для кластера {cluster_id}: {e}")
                    similar_words[cluster_id] = []
            else:
                similar_words[cluster_id] = []
        
        return similar_words
    
    def visualize_clusters_umap(self, n_components: int = 2,
                               n_neighbors: int = 15,
                               min_dist: float = 0.1,
                               save_path: Optional[str] = None) -> np.ndarray:
        """Визуализация кластеров с помощью UMAP."""
        if self.vectors is None or self.labels is None:
            raise ValueError("Данные не установлены. Вызовите set_data() сначала.")
        
        if not UMAP_AVAILABLE or UMAP is None:
            raise ImportError("UMAP не установлен. Установите: pip install umap-learn")
        
        logger.info("Применение UMAP для снижения размерности...")
        
        # Применяем UMAP
        reducer = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        
        embedding = reducer.fit_transform(self.vectors)
        
        # Визуализация
        plt.figure(figsize=(12, 8))
        
        unique_labels = set(self.labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
            # Отдельно рисуем шум
            noise_mask = self.labels == -1
            if noise_mask.any():
                plt.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                           c='gray', s=10, alpha=0.3, label='Шум')
        
        # Рисуем кластеры
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(sorted(unique_labels)):
            mask = self.labels == label
            plt.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[colors[i]], s=20, alpha=0.6, label=f'Кластер {label}')
        
        plt.title('Визуализация кластеров (UMAP)', fontsize=14)
        plt.xlabel('UMAP 1', fontsize=12)
        plt.ylabel('UMAP 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Визуализация сохранена: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return embedding
    
    def visualize_clusters_pca(self, n_components: int = 2,
                              save_path: Optional[str] = None) -> np.ndarray:
        """Визуализация кластеров с помощью PCA."""
        if self.vectors is None or self.labels is None:
            raise ValueError("Данные не установлены. Вызовите set_data() сначала.")
        
        logger.info("Применение PCA для снижения размерности...")
        
        # Применяем PCA
        reducer = PCA(n_components=n_components, random_state=42)
        embedding = reducer.fit_transform(self.vectors)
        
        explained_variance = reducer.explained_variance_ratio_.sum()
        logger.info(f"Объясненная дисперсия: {explained_variance:.4f}")
        
        # Визуализация
        plt.figure(figsize=(12, 8))
        
        unique_labels = set(self.labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
            noise_mask = self.labels == -1
            if noise_mask.any():
                plt.scatter(embedding[noise_mask, 0], embedding[noise_mask, 1],
                           c='gray', s=10, alpha=0.3, label='Шум')
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        for i, label in enumerate(sorted(unique_labels)):
            mask = self.labels == label
            plt.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[colors[i]], s=20, alpha=0.6, label=f'Кластер {label}')
        
        plt.title(f'Визуализация кластеров (PCA, объясненная дисперсия: {explained_variance:.2%})', 
                 fontsize=14)
        plt.xlabel('PC1', fontsize=12)
        plt.ylabel('PC2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Визуализация сохранена: {save_path}")
        else:
            plt.show()
        
        plt.close()
        
        return embedding
    
    def print_cluster_summary(self, top_words: Dict[int, List[Tuple[str, Any]]],
                             cluster_sizes: Optional[Dict[int, int]] = None):
        """Вывод сводки по кластерам."""
        print("\n" + "="*70)
        print("СВОДКА ПО КЛАСТЕРАМ")
        print("="*70)
        
        for cluster_id in sorted(top_words.keys()):
            size = cluster_sizes.get(cluster_id, 0) if cluster_sizes else 0
            print(f"\nКластер {cluster_id} (размер: {size} документов):")
            print("-" * 70)
            
            for i, (word, value) in enumerate(top_words[cluster_id], 1):
                if isinstance(value, float):
                    print(f"  {i:2d}. {word:20s} (TF-IDF: {value:.4f})")
                else:
                    print(f"  {i:2d}. {word:20s} (частота: {value})")
        
        print("="*70 + "\n")

