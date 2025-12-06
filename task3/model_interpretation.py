"""
Модуль для интерпретации и визуализации моделей.
Этап 7: Интерпретация и визуализация моделей.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Интерпретация
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_text
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    """Класс для анализа важности признаков."""
    
    def __init__(self, model: Any, vectorizer: Any = None):
        self.model = model
        self.vectorizer = vectorizer
    
    def get_linear_weights(self) -> Dict[str, float]:
        """Извлечение весов для линейных моделей."""
        if not hasattr(self.model, 'coef_'):
            return {}
        
        if self.vectorizer is None:
            return {}
        
        feature_names = self.vectorizer.get_feature_names_out()
        coef = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
        
        importance = dict(zip(feature_names, coef))
        return dict(sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True))
    
    def get_tree_importance(self) -> Dict[str, float]:
        """Извлечение важности признаков для tree-based моделей."""
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        if self.vectorizer is None:
            return {}
        
        feature_names = self.vectorizer.get_feature_names_out()
        importances = self.model.feature_importances_
        
        importance = dict(zip(feature_names, importances))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def get_top_features(self, n: int = 20, method: str = 'auto') -> Dict[str, float]:
        """Получение топ-N важных признаков."""
        if method == 'auto':
            if hasattr(self.model, 'coef_'):
                importance = self.get_linear_weights()
            elif hasattr(self.model, 'feature_importances_'):
                importance = self.get_tree_importance()
            else:
                return {}
        elif method == 'linear':
            importance = self.get_linear_weights()
        elif method == 'tree':
            importance = self.get_tree_importance()
        else:
            return {}
        
        return dict(list(importance.items())[:n])


class SHAPExplainer:
    """Класс для SHAP объяснений."""
    
    def __init__(self, model: Any, X: np.ndarray, vectorizer: Any = None):
        self.model = model
        self.X = X
        self.vectorizer = vectorizer
        self.explainer = None
        self.shap_values = None
    
    def explain(self, X_explain: Optional[np.ndarray] = None, max_evals: int = 100):
        """Вычисление SHAP значений."""
        if not SHAP_AVAILABLE:
            logger.warning("SHAP не доступен")
            return None
        
        # Проверяем, поддерживает ли модель SHAP
        from sklearn.ensemble import VotingClassifier
        if isinstance(self.model, VotingClassifier):
            logger.warning("SHAP не поддерживается для VotingClassifier (stacking/voting)")
            return None
        
        if X_explain is None:
            X_explain = self.X[:min(100, len(self.X))]  # Ограничиваем для скорости
        
        # Ограничиваем размер background dataset для производительности
        background_size = min(50, len(self.X))
        X_background = self.X[:background_size] if len(self.X) > 0 else self.X
        
        if len(X_background) == 0:
            logger.warning("Background dataset пуст")
            return None
        
        try:
            # Пытаемся использовать TreeExplainer для tree-based моделей
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.tree import DecisionTreeClassifier
            
            tree_models = [RandomForestClassifier, GradientBoostingClassifier, DecisionTreeClassifier]
            
            # Пытаемся импортировать дополнительные библиотеки, если они доступны
            try:
                from xgboost import XGBClassifier
                tree_models.append(XGBClassifier)
            except ImportError:
                pass
            
            try:
                from lightgbm import LGBMClassifier
                tree_models.append(LGBMClassifier)
            except ImportError:
                pass
            
            try:
                from catboost import CatBoostClassifier
                tree_models.append(CatBoostClassifier)
            except ImportError:
                pass
            
            tree_models = tuple(tree_models)
            
            if isinstance(self.model, tree_models):
                try:
                    self.explainer = shap.TreeExplainer(self.model)
                    shap_vals = self.explainer.shap_values(X_explain)
                    # TreeExplainer может вернуть список для многоклассовой классификации
                    # Преобразуем в массив для единообразия
                    if isinstance(shap_vals, list):
                        # Для многоклассовой классификации возвращаем список
                        self.shap_values = shap_vals
                    else:
                        # Для бинарной классификации возвращаем массив
                        self.shap_values = shap_vals
                    logger.info("SHAP значения вычислены (TreeExplainer)")
                    return self.shap_values
                except Exception as e:
                    logger.warning(f"TreeExplainer не сработал: {e}, пробуем Explainer")
            
            # Для других моделей используем Explainer
            try:
                self.explainer = shap.Explainer(self.model, X_background)
                self.shap_values = self.explainer(X_explain)
                logger.info("SHAP значения вычислены")
                return self.shap_values
            except Exception as e:
                # Если Explainer не работает, пробуем KernelExplainer
                try:
                    logger.info("Пробуем KernelExplainer...")
                    self.explainer = shap.KernelExplainer(self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict, X_background)
                    self.shap_values = self.explainer.shap_values(X_explain)
                    logger.info("SHAP значения вычислены (KernelExplainer)")
                    return self.shap_values
                except Exception as e2:
                    logger.error(f"Ошибка вычисления SHAP (KernelExplainer): {e2}")
                    return None
        except Exception as e:
            logger.error(f"Ошибка вычисления SHAP: {e}")
            return None
    
    def plot_summary(self, save_path: Optional[str] = None):
        """Визуализация summary plot."""
        if self.shap_values is None:
            logger.warning("SHAP значения не вычислены")
            return
        
        try:
            shap.summary_plot(self.shap_values, show=False)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Ошибка визуализации SHAP: {e}")
    
    def plot_waterfall(self, instance_idx: int = 0, save_path: Optional[str] = None):
        """Визуализация waterfall plot для одного примера."""
        if self.shap_values is None:
            logger.warning("SHAP значения не вычислены")
            return
        
        try:
            shap.waterfall_plot(self.shap_values[instance_idx], show=False)
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.error(f"Ошибка визуализации waterfall: {e}")


class LIMEExplainer:
    """Класс для LIME объяснений."""
    
    def __init__(self, model: Any, texts: List[str], class_names: List[str] = None):
        self.model = model
        self.texts = texts
        self.class_names = class_names or ['negative', 'positive']
        self.explainer = None
        
        if LIME_AVAILABLE:
            self.explainer = LimeTextExplainer(class_names=self.class_names)
    
    def explain_instance(self, text: str, num_features: int = 10, predict_proba_func: Any = None) -> Optional[Any]:
        """Объяснение одного примера."""
        if not LIME_AVAILABLE or self.explainer is None:
            logger.warning("LIME не доступен")
            return None
        
        # Используем переданную функцию предсказания, если она есть
        if predict_proba_func is None:
            def predict_proba(texts):
                # Преобразуем тексты в векторы (упрощенная версия)
                # В реальности нужно использовать тот же векторизатор, что и при обучении
                predictions = []
                for t in texts:
                    # Здесь должна быть векторизация и предсказание
                    # Для демонстрации возвращаем случайные значения
                    if hasattr(self.model, 'predict_proba'):
                        try:
                            # Пытаемся вызвать predict_proba напрямую (если модель принимает тексты)
                            pred = self.model.predict_proba([t])[0]
                        except:
                            # Если не работает, возвращаем равномерное распределение
                            num_classes = len(self.class_names) if self.class_names else 2
                            pred = np.array([1.0/num_classes] * num_classes)
                    else:
                        num_classes = len(self.class_names) if self.class_names else 2
                        pred = np.array([1.0/num_classes] * num_classes)
                    predictions.append(pred)
                return np.array(predictions)
            predict_proba_func = predict_proba
        
        try:
            explanation = self.explainer.explain_instance(
                text, predict_proba_func, num_features=num_features
            )
            return explanation
        except Exception as e:
            logger.error(f"Ошибка LIME объяснения: {e}")
            return None


class VisualizationTools:
    """Класс для визуализации результатов."""
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             class_names: Optional[List[str]] = None, save_path: Optional[str] = None):
        """Визуализация матрицы ошибок."""
        cm = confusion_matrix(y_true, y_pred)
        
        # Если class_names не указаны, используем числовые метки
        if class_names is None:
            unique_labels = sorted(set(y_true) | set(y_pred))
            class_names = [str(label) for label in unique_labels]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.ylabel('Истинные значения')
        plt.xlabel('Предсказанные значения')
        plt.title('Матрица ошибок')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                      save_path: Optional[str] = None):
        """Визуализация ROC-кривой."""
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = np.trapz(tpr, fpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Случайный классификатор')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC-кривая')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pr_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                     save_path: Optional[str] = None):
        """Визуализация Precision-Recall кривой."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        pr_auc = np.trapz(precision, recall)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall кривая')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_embeddings(self, embeddings: np.ndarray, labels: np.ndarray,
                       method: str = 'tsne', n_components: int = 2,
                       save_path: Optional[str] = None):
        """Визуализация эмбеддингов с помощью t-SNE или UMAP."""
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
            reduced = reducer.fit_transform(embeddings)
        elif method == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(embeddings)
        elif method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(embeddings)
        else:
            logger.warning(f"Метод {method} не доступен, используем PCA")
            reducer = PCA(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'Визуализация эмбеддингов ({method.upper()})')
        plt.xlabel('Компонента 1')
        plt.ylabel('Компонента 2')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, importance: Dict[str, float], top_n: int = 20,
                               save_path: Optional[str] = None):
        """Визуализация важности признаков."""
        top_features = dict(list(importance.items())[:top_n])
        
        plt.figure(figsize=(10, 8))
        features = list(top_features.keys())
        values = list(top_features.values())
        
        plt.barh(features, values)
        plt.xlabel('Важность')
        plt.title(f'Топ-{top_n} важных признаков')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]],
                             metric: str = 'f1_score', save_path: Optional[str] = None):
        """Сравнение моделей по метрике."""
        models = list(results.keys())
        scores = [results[m].get(metric, 0) for m in models]
        
        plt.figure(figsize=(12, 6))
        plt.bar(models, scores)
        plt.ylabel(metric)
        plt.title(f'Сравнение моделей по метрике {metric}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


class ErrorAnalysis:
    """Класс для анализа ошибок."""
    
    def __init__(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, texts: List[str]):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.texts = texts
        self.errors = []
    
    def find_errors(self, n_errors: int = 20) -> List[Dict]:
        """Поиск примеров с ошибками."""
        y_pred = self.model.predict(self.X_test)
        
        # Проверяем, является ли это multilabel задачей (матрица вместо вектора)
        is_multilabel = len(self.y_test.shape) > 1 and self.y_test.shape[1] > 1
        
        if is_multilabel:
            # Для multilabel сравниваем по каждому классу
            # Находим индексы, где предсказания не совпадают с истинными метками
            differences = np.any(y_pred != self.y_test, axis=1)
            error_indices = np.where(differences)[0]
            
            errors = []
            for idx in error_indices[:n_errors]:
                # Для multilabel сохраняем списки меток
                true_labels = self.y_test[idx].tolist()
                pred_labels = y_pred[idx].tolist()
                errors.append({
                    'index': int(idx),
                    'text': self.texts[idx],
                    'true_label': [int(l) for l in true_labels],  # Список меток
                    'predicted_label': [int(l) for l in pred_labels]  # Список меток
                })
        else:
            # Для binary/multiclass - обычная обработка
            error_indices = np.where(y_pred != self.y_test)[0]
            
            errors = []
            for idx in error_indices[:n_errors]:
                errors.append({
                    'index': int(idx),
                    'text': self.texts[idx],
                    'true_label': int(self.y_test[idx]),  # Преобразуем numpy тип в int
                    'predicted_label': int(y_pred[idx])   # Преобразуем numpy тип в int
                })
        
        self.errors = errors
        return errors
    
    def analyze_error_patterns(self) -> Dict[str, Any]:
        """Анализ паттернов ошибок."""
        if not self.errors:
            self.find_errors()
        
        # Группировка по типам ошибок
        error_types = {}
        for error in self.errors:
            # Преобразуем метки в строки для ключа
            if isinstance(error['true_label'], list):
                # Для multilabel - преобразуем список в строку
                true_label = str(error['true_label'])
                pred_label = str(error['predicted_label'])
            else:
                # Для binary/multiclass
                true_label = str(int(error['true_label'])) if isinstance(error['true_label'], (np.integer, np.int64)) else str(error['true_label'])
                pred_label = str(int(error['predicted_label'])) if isinstance(error['predicted_label'], (np.integer, np.int64)) else str(error['predicted_label'])
            
            error_type = f"{true_label} -> {pred_label}"
            if error_type not in error_types:
                error_types[error_type] = []
            # Убеждаемся, что все значения сериализуемы
            if isinstance(error['true_label'], list):
                serializable_error = {
                    'index': int(error['index']),
                    'text': str(error['text']),
                    'true_label': error['true_label'],  # Уже список
                    'predicted_label': error['predicted_label']  # Уже список
                }
            else:
                serializable_error = {
                    'index': int(error['index']),
                    'text': str(error['text']),
                    'true_label': int(error['true_label']) if isinstance(error['true_label'], (np.integer, np.int64)) else error['true_label'],
                    'predicted_label': int(error['predicted_label']) if isinstance(error['predicted_label'], (np.integer, np.int64)) else error['predicted_label']
                }
            error_types[error_type].append(serializable_error)
        
        return {
            'total_errors': int(len(self.errors)),
            'error_types': {k: int(len(v)) for k, v in error_types.items()},
            'examples': error_types
        }

