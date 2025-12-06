"""
Модуль для борьбы с дисбалансом классов.
Этап 5: Методы борьбы с дисбалансом классов в текстовых данных.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Any, Optional, TYPE_CHECKING
from collections import Counter
import random

# Условный импорт для аннотаций типов
if TYPE_CHECKING:
    try:
        import torch
    except ImportError:
        torch = None

# PyTorch для весов классов
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    torch = None

# Библиотеки для сэмплирования
try:
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# Библиотеки для аугментации
try:
    import nlpaug.augmenter.word as naw
    NLPAUG_AVAILABLE = True
except ImportError:
    NLPAUG_AVAILABLE = False

# Transformers для back translation (опционально, может быть недоступен в некоторых регионах)
TRANSFORMERS_AVAILABLE = False
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

# Альтернативные библиотеки для перевода (если Hugging Face недоступен)
try:
    from googletrans import Translator
    GOOGLETRANS_AVAILABLE = True
except ImportError:
    GOOGLETRANS_AVAILABLE = False

try:
    import deep_translator
    DEEP_TRANSLATOR_AVAILABLE = True
except ImportError:
    DEEP_TRANSLATOR_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassWeightCalculator:
    """Класс для расчета весов классов."""
    
    @staticmethod
    def compute_sklearn_weights(y: np.ndarray) -> Dict[int, float]:
        """Вычисление весов классов для scikit-learn."""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    
    @staticmethod
    def compute_torch_weights(y: np.ndarray) -> Any:
        """Вычисление весов классов для PyTorch."""
        if not PYTORCH_AVAILABLE or torch is None:
            raise ImportError("PyTorch не установлен")
        
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return torch.FloatTensor(weights)


class SamplingMethods:
    """Класс для методов сэмплирования."""
    
    def __init__(self):
        self.available = IMBLEARN_AVAILABLE
    
    def random_oversample(self, X: np.ndarray, y: np.ndarray,
                         sampling_strategy: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        """Случайная перевыборка."""
        if not self.available:
            logger.warning("imbalanced-learn не доступен")
            return X, y
        
        sampler = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        logger.info(f"Random Oversampling: {len(y)} -> {len(y_resampled)}")
        return X_resampled, y_resampled
    
    def random_undersample(self, X: np.ndarray, y: np.ndarray,
                          sampling_strategy: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        """Случайная недовыборка."""
        if not self.available:
            logger.warning("imbalanced-learn не доступен")
            return X, y
        
        sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        logger.info(f"Random Undersampling: {len(y)} -> {len(y_resampled)}")
        return X_resampled, y_resampled
    
    def smote(self, X: np.ndarray, y: np.ndarray,
             sampling_strategy: str = 'auto', k_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """SMOTE - синтез новых примеров миноритарного класса."""
        if not self.available:
            logger.warning("imbalanced-learn не доступен")
            return X, y
        
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_neighbors,
            random_state=42
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        logger.info(f"SMOTE: {len(y)} -> {len(y_resampled)}")
        return X_resampled, y_resampled
    
    def adasyn(self, X: np.ndarray, y: np.ndarray,
              sampling_strategy: str = 'auto', n_neighbors: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """ADASYN - адаптивный синтез новых примеров."""
        if not self.available:
            logger.warning("imbalanced-learn не доступен")
            return X, y
        
        sampler = ADASYN(
            sampling_strategy=sampling_strategy,
            n_neighbors=n_neighbors,
            random_state=42
        )
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        logger.info(f"ADASYN: {len(y)} -> {len(y_resampled)}")
        return X_resampled, y_resampled


class TextAugmentation:
    """Класс для аугментации текстовых данных."""
    
    def __init__(self, language: str = 'ru'):
        self.language = language
        self.synonym_aug = None
        self.back_translation_pipeline = None
        
        # Инициализация nlpaug
        if NLPAUG_AVAILABLE:
            try:
                if language == 'ru':
                    # Для русского языка используем WordNet или словари
                    self.synonym_aug = naw.SynonymAug(aug_src='wordnet', lang='rus')
                else:
                    self.synonym_aug = naw.SynonymAug(aug_src='wordnet')
            except:
                logger.warning("Не удалось инициализировать SynonymAug")
        
        # Инициализация back translation (НЕ загружается автоматически)
        # Поддержка нескольких методов перевода для обхода блокировок
        self.back_translation_pipeline = None
        self.back_translation_translator = None
        self.back_translation_method = None
        self.language = language
        
        # Определяем доступный метод перевода
        if TRANSFORMERS_AVAILABLE:
            self.back_translation_model_name = (
                f"Helsinki-NLP/opus-mt-{language}-en" if language == 'ru' 
                else f"Helsinki-NLP/opus-mt-en-{language}"
            )
        elif GOOGLETRANS_AVAILABLE:
            try:
                self.back_translation_translator = Translator()
                self.back_translation_method = 'googletrans'
            except:
                pass
        elif DEEP_TRANSLATOR_AVAILABLE:
            self.back_translation_method = 'deep_translator'
    
    def synonym_replacement(self, text: str, num_replacements: int = 1) -> str:
        """Замена синонимами."""
        if self.synonym_aug is None:
            return text
        
        try:
            augmented = self.synonym_aug.augment(text, num_replacements)
            return augmented if isinstance(augmented, str) else augmented[0]
        except:
            return text
    
    def word_insertion(self, text: str, num_insertions: int = 1) -> str:
        """Вставка слов."""
        if not NLPAUG_AVAILABLE:
            return text
        
        try:
            aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="insert")
            augmented = aug.augment(text, num_insertions)
            return augmented if isinstance(augmented, str) else augmented[0]
        except:
            return text
    
    def word_deletion(self, text: str, p: float = 0.1) -> str:
        """Удаление слов."""
        if not NLPAUG_AVAILABLE:
            return text
        
        try:
            aug = naw.RandomWordAug(action="delete", aug_p=p)
            augmented = aug.augment(text)
            return augmented if isinstance(augmented, str) else augmented[0]
        except:
            return text
    
    def word_swap(self, text: str, num_swaps: int = 1) -> str:
        """Перестановка слов."""
        if not NLPAUG_AVAILABLE:
            return text
        
        try:
            aug = naw.RandomWordAug(action="swap", aug_p=0.1)
            augmented = aug.augment(text)
            return augmented if isinstance(augmented, str) else augmented[0]
        except:
            return text
    
    def back_translation(self, text: str, load_model: bool = False) -> str:
        """
        Back Translation (перевод на другой язык и обратно).
        Поддерживает несколько методов для обхода блокировок.
        
        Args:
            text: Текст для аугментации
            load_model: Если True, попытается загрузить модель при первом вызове
        
        Returns:
            Аугментированный текст или исходный текст, если перевод недоступен
        """
        # Метод 1: Hugging Face Transformers (может быть заблокирован)
        if TRANSFORMERS_AVAILABLE and load_model:
            if self.back_translation_pipeline is None:
                try:
                    logger.info(f"Попытка загрузки модели Hugging Face: {self.back_translation_model_name}")
                    logger.info("Если сайт заблокирован, будут использованы альтернативные методы...")
                    self.back_translation_pipeline = pipeline(
                        "translation",
                        model=self.back_translation_model_name,
                        device=-1
                    )
                    self.back_translation_method = 'transformers'
                    logger.info("Модель Hugging Face загружена успешно")
                except Exception as e:
                    logger.warning(f"Hugging Face недоступен: {e}")
                    logger.info("Попытка использовать альтернативные методы перевода...")
                    self.back_translation_pipeline = None
            
            if self.back_translation_pipeline is not None:
                try:
                    translated = self.back_translation_pipeline(text)[0]['translation_text']
                    back_translated = self.back_translation_pipeline(translated)[0]['translation_text']
                    return back_translated
                except Exception as e:
                    logger.debug(f"Ошибка Hugging Face перевода: {e}")
        
        # Метод 2: Google Translate (через googletrans)
        if GOOGLETRANS_AVAILABLE and (self.back_translation_translator is not None or load_model):
            try:
                if self.back_translation_translator is None:
                    self.back_translation_translator = Translator()
                    self.back_translation_method = 'googletrans'
                
                # Переводим на английский
                if self.language == 'ru':
                    translated = self.back_translation_translator.translate(text, src='ru', dest='en').text
                    # Переводим обратно на русский
                    back_translated = self.back_translation_translator.translate(translated, src='en', dest='ru').text
                else:
                    translated = self.back_translation_translator.translate(text, src=self.language, dest='en').text
                    back_translated = self.back_translation_translator.translate(translated, src='en', dest=self.language).text
                
                return back_translated
            except Exception as e:
                logger.debug(f"Ошибка Google Translate: {e}")
        
        # Метод 3: Deep Translator (альтернативный сервис)
        if DEEP_TRANSLATOR_AVAILABLE and load_model:
            try:
                from deep_translator import GoogleTranslator
                
                if self.language == 'ru':
                    # Переводим на английский
                    translated = GoogleTranslator(source='ru', target='en').translate(text)
                    # Переводим обратно
                    back_translated = GoogleTranslator(source='en', target='ru').translate(translated)
                else:
                    translated = GoogleTranslator(source=self.language, target='en').translate(text)
                    back_translated = GoogleTranslator(source='en', target=self.language).translate(translated)
                
                self.back_translation_method = 'deep_translator'
                return back_translated
            except Exception as e:
                logger.debug(f"Ошибка Deep Translator: {e}")
        
        # Если все методы недоступны
        logger.warning("Back translation недоступен. Все методы перевода не работают.")
        logger.info("Рекомендуется использовать другие методы аугментации: 'synonym' или 'eda'")
        return text
    
    def easy_data_augmentation(self, text: str, num_aug: int = 4) -> List[str]:
        """Easy Data Augmentation (EDA)."""
        augmented_texts = []
        
        # Синонимы
        augmented_texts.append(self.synonym_replacement(text))
        
        # Вставка
        augmented_texts.append(self.word_insertion(text))
        
        # Удаление
        augmented_texts.append(self.word_deletion(text))
        
        # Перестановка
        augmented_texts.append(self.word_swap(text))
        
        return augmented_texts[:num_aug]
    
    def augment_texts(self, texts: List[str], labels: List[Any],
                     target_class: Any, num_augmentations: int = 1,
                     method: str = 'eda') -> Tuple[List[str], List[Any]]:
        """Аугментация текстов для определенного класса."""
        augmented_texts = []
        augmented_labels = []
        
        # Находим тексты целевого класса
        target_texts = [text for text, label in zip(texts, labels) if label == target_class]
        
        if not target_texts:
            return texts, labels
        
        logger.info(f"Аугментация {len(target_texts)} текстов класса {target_class} методом {method}")
        
        for text in target_texts:
            if method == 'eda':
                aug_texts = self.easy_data_augmentation(text, num_augmentations)
            elif method == 'synonym':
                aug_texts = [self.synonym_replacement(text) for _ in range(num_augmentations)]
            elif method == 'back_translation':
                # Back translation требует явной загрузки модели
                # Может быть недоступен в некоторых регионах (например, Россия)
                # В этом случае рекомендуется использовать 'eda' или 'synonym'
                aug_texts = [self.back_translation(text, load_model=True) for _ in range(num_augmentations)]
                # Фильтруем пустые результаты (если перевод не удался)
                aug_texts = [t for t in aug_texts if t and t != text]
                if not aug_texts:
                    logger.warning("Back translation недоступен. Используйте метод 'eda' вместо 'back_translation'")
                    # Fallback на EDA
                    aug_texts = self.easy_data_augmentation(text, num_augmentations)
            else:
                aug_texts = [text]
            
            augmented_texts.extend(aug_texts)
            augmented_labels.extend([target_class] * len(aug_texts))
        
        # Объединяем с исходными данными
        all_texts = texts + augmented_texts
        all_labels = labels + augmented_labels
        
        return all_texts, all_labels


class ImbalanceHandler:
    """Главный класс для обработки дисбаланса классов."""
    
    def __init__(self, language: str = 'ru'):
        self.sampling = SamplingMethods()
        self.augmentation = TextAugmentation(language)
        self.class_weights = ClassWeightCalculator()
    
    def handle_imbalance(self, X: np.ndarray, y: np.ndarray, texts: List[str],
                       method: str = 'smote', target_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обработка дисбаланса классов.
        
        Args:
            X: Признаки (векторы)
            y: Метки
            texts: Исходные тексты (для аугментации)
            method: Метод ('smote', 'adasyn', 'oversample', 'undersample', 'augmentation')
            target_ratio: Целевое соотношение классов
        
        Returns:
            X_resampled, y_resampled
        """
        logger.info(f"Обработка дисбаланса методом: {method}")
        
        if method == 'smote':
            return self.sampling.smote(X, y)
        elif method == 'adasyn':
            return self.sampling.adasyn(X, y)
        elif method == 'oversample':
            return self.sampling.random_oversample(X, y)
        elif method == 'undersample':
            return self.sampling.random_undersample(X, y)
        elif method == 'augmentation':
            # Аугментация текстов
            class_counts = Counter(y)
            minority_class = min(class_counts, key=class_counts.get)
            majority_count = max(class_counts.values())
            minority_count = class_counts[minority_class]
            
            num_augmentations = (majority_count - minority_count) // minority_count
            
            augmented_texts, augmented_labels = self.augmentation.augment_texts(
                texts, y.tolist(), minority_class, num_augmentations, method='eda'
            )
            
            # Нужно пересоздать X для новых текстов
            # Это упрощенная версия - в реальности нужно перевекторизовать
            logger.warning("Аугментация текстов требует перевекторизации признаков")
            return X, y
        else:
            logger.warning(f"Неизвестный метод: {method}")
            return X, y
    
    def get_class_weights(self, y: np.ndarray, framework: str = 'sklearn') -> Any:
        """Получение весов классов."""
        if framework == 'sklearn':
            return self.class_weights.compute_sklearn_weights(y)
        elif framework == 'pytorch':
            return self.class_weights.compute_torch_weights(y)
        else:
            raise ValueError(f"Неизвестный фреймворк: {framework}")

