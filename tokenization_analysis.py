import os
os.environ['TCL_LIBRARY'] = "C:/Program Files/Python313/tcl/tcl8.6"
os.environ['TK_LIBRARY'] = "C:/Program Files/Python313/tcl/tk8.6"
import re
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import logging
from collections import Counter
import json
import os

# Импорт библиотек для токенизации
try:
    import nltk
    from nltk.tokenize import word_tokenize, regexp_tokenize
    from nltk.stem import PorterStemmer, SnowballStemmer
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("NLTK не установлен. Установите: pip install nltk")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("spaCy не установлен. Установите: pip install spacy")

try:
    import razdel
    RAZDEL_AVAILABLE = True
except ImportError:
    RAZDEL_AVAILABLE = False
    print("razdel не установлен. Установите: pip install razdel")

try:
    import pymorphy2
    PYMORPHY_AVAILABLE = True
except ImportError:
    PYMORPHY_AVAILABLE = False
    print("pymorphy2 не установлен. Установите: pip install pymorphy2")

logger = logging.getLogger(__name__)

class TokenizationAnalyzer:
    """Анализатор методов токенизации и нормализации"""
    
    def __init__(self, language: str = 'russian'):
        self.language = language
        self.results = {}
        
        # Инициализация инструментов
        self._init_tools()
        
        # Загрузка стоп-слов
        self.stopwords = self._load_stopwords()
    
    def _init_tools(self):
        """Инициализация инструментов токенизации"""
        # NLTK
        if NLTK_AVAILABLE:
            try:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                self.nltk_stemmer = PorterStemmer()
                self.nltk_snowball = SnowballStemmer('russian')
            except Exception as e:
                logger.warning(f"Ошибка инициализации NLTK: {e}")
        
        # spaCy
        if SPACY_AVAILABLE:
            try:
                self.spacy_nlp = spacy.load("ru_core_news_sm")
            except OSError:
                logger.warning("Модель ru_core_news_sm не найдена. Установите: python -m spacy download ru_core_news_sm")
                self.spacy_nlp = None
        
        # pymorphy2
        if PYMORPHY_AVAILABLE:
            try:
                self.pymorphy_analyzer = pymorphy2.MorphAnalyzer()
            except Exception as e:
                logger.warning(f"Ошибка инициализации pymorphy2: {e}")
                self.pymorphy_analyzer = None
    
    def _load_stopwords(self) -> set:
        """Загрузка стоп-слов"""
        stopwords_set = set()
        
        if NLTK_AVAILABLE:
            try:
                nltk_stopwords = set(stopwords.words('russian'))
                stopwords_set.update(nltk_stopwords)
            except Exception as e:
                logger.warning(f"Ошибка загрузки стоп-слов NLTK: {e}")
        
        # Дополнительные русские стоп-слова
        russian_stopwords = {
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'
        }
        stopwords_set.update(russian_stopwords)
        
        return stopwords_set
    
    def naive_tokenization(self, text: str) -> List[str]:
        """Наивная токенизация по пробелам"""
        return text.split()
    
    def regex_tokenization(self, text: str) -> List[str]:
        """Токенизация на основе регулярных выражений"""
        if not NLTK_AVAILABLE:
            return self.naive_tokenization(text)
        
        # Паттерн для токенизации слов
        pattern = r'\b\w+\b'
        return regexp_tokenize(text, pattern)
    
    def nltk_tokenization(self, text: str) -> List[str]:
        """Токенизация с помощью NLTK"""
        if not NLTK_AVAILABLE:
            return self.naive_tokenization(text)
        
        try:
            return word_tokenize(text, language='russian')
        except Exception as e:
            logger.warning(f"Ошибка NLTK токенизации: {e}")
            return self.naive_tokenization(text)
    
    def spacy_tokenization(self, text: str) -> List[str]:
        """Токенизация с помощью spaCy"""
        if not SPACY_AVAILABLE or self.spacy_nlp is None:
            return self.naive_tokenization(text)
        
        try:
            doc = self.spacy_nlp(text)
            return [token.text for token in doc]
        except Exception as e:
            logger.warning(f"Ошибка spaCy токенизации: {e}")
            return self.naive_tokenization(text)
    
    def razdel_tokenization(self, text: str) -> List[str]:
        """Токенизация с помощью razdel"""
        if not RAZDEL_AVAILABLE:
            return self.naive_tokenization(text)
        
        try:
            tokens = list(razdel.tokenize(text))
            return [token.text for token in tokens]
        except Exception as e:
            logger.warning(f"Ошибка razdel токенизации: {e}")
            return self.naive_tokenization(text)
    
    def porter_stemming(self, tokens: List[str]) -> List[str]:
        """Стемминг с помощью Porter Stemmer"""
        if not NLTK_AVAILABLE:
            return tokens
        
        try:
            return [self.nltk_stemmer.stem(token) for token in tokens]
        except Exception as e:
            logger.warning(f"Ошибка Porter стемминга: {e}")
            return tokens
    
    def snowball_stemming(self, tokens: List[str]) -> List[str]:
        """Стемминг с помощью Snowball Stemmer"""
        if not NLTK_AVAILABLE:
            return tokens
        
        try:
            return [self.nltk_snowball.stem(token) for token in tokens]
        except Exception as e:
            logger.warning(f"Ошибка Snowball стемминга: {e}")
            return tokens
    
    def spacy_lemmatization(self, tokens: List[str]) -> List[str]:
        """Лемматизация с помощью spaCy"""
        if not SPACY_AVAILABLE or self.spacy_nlp is None:
            return tokens
        
        try:
            text = ' '.join(tokens)
            doc = self.spacy_nlp(text)
            return [token.lemma_ for token in doc]
        except Exception as e:
            logger.warning(f"Ошибка spaCy лемматизации: {e}")
            return tokens
    
    def pymorphy_lemmatization(self, tokens: List[str]) -> List[str]:
        """Лемматизация с помощью pymorphy2"""
        if not PYMORPHY_AVAILABLE or self.pymorphy_analyzer is None:
            return tokens
        
        try:
            lemmas = []
            for token in tokens:
                parsed = self.pymorphy_analyzer.parse(token)[0]
                lemmas.append(parsed.normal_form)
            return lemmas
        except Exception as e:
            logger.warning(f"Ошибка pymorphy2 лемматизации: {e}")
            return tokens
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Удаление стоп-слов"""
        return [token for token in tokens if token.lower() not in self.stopwords]
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Анализ текста различными методами"""
        results = {}
        
        # Методы токенизации
        tokenization_methods = {
            'naive': self.naive_tokenization,
            'regex': self.regex_tokenization,
            'nltk': self.nltk_tokenization,
            'spacy': self.spacy_tokenization,
            'razdel': self.razdel_tokenization
        }
        
        for method_name, method_func in tokenization_methods.items():
            try:
                start_time = time.time()
                tokens = method_func(text)
                processing_time = time.time() - start_time
                
                # Базовые метрики
                results[method_name] = {
                    'tokens': tokens,
                    'token_count': len(tokens),
                    'unique_tokens': len(set(tokens)),
                    'vocabulary_size': len(set(tokens)),
                    'processing_time': processing_time,
                    'avg_token_length': np.mean([len(token) for token in tokens]) if tokens else 0
                }
                
            except Exception as e:
                logger.error(f"Ошибка в методе {method_name}: {e}")
                results[method_name] = {
                    'tokens': [],
                    'token_count': 0,
                    'unique_tokens': 0,
                    'vocabulary_size': 0,
                    'processing_time': 0,
                    'avg_token_length': 0,
                    'error': str(e)
                }
        
        return results
    
    def analyze_normalization(self, tokens: List[str]) -> Dict[str, Any]:
        """Анализ методов нормализации"""
        results = {}
        
        # Методы нормализации
        normalization_methods = {
            'original': lambda x: x,
            'porter_stem': self.porter_stemming,
            'snowball_stem': self.snowball_stemming,
            'spacy_lemma': self.spacy_lemmatization,
            'pymorphy_lemma': self.pymorphy_lemmatization,
            'no_stopwords': self.remove_stopwords
        }
        
        for method_name, method_func in normalization_methods.items():
            try:
                start_time = time.time()
                normalized_tokens = method_func(tokens)
                processing_time = time.time() - start_time
                
                results[method_name] = {
                    'tokens': normalized_tokens,
                    'token_count': len(normalized_tokens),
                    'unique_tokens': len(set(normalized_tokens)),
                    'vocabulary_size': len(set(normalized_tokens)),
                    'processing_time': processing_time,
                    'compression_ratio': len(tokens) / len(normalized_tokens) if normalized_tokens else 1
                }
                
            except Exception as e:
                logger.error(f"Ошибка в методе нормализации {method_name}: {e}")
                results[method_name] = {
                    'tokens': [],
                    'token_count': 0,
                    'unique_tokens': 0,
                    'vocabulary_size': 0,
                    'processing_time': 0,
                    'compression_ratio': 1,
                    'error': str(e)
                }
        
        return results
    
    def calculate_oov_rate(self, train_tokens: List[str], test_tokens: List[str]) -> float:
        """Расчет доли OOV (Out-of-Vocabulary) токенов"""
        train_vocab = set(train_tokens)
        test_vocab = set(test_tokens)
        
        oov_tokens = test_vocab - train_vocab
        oov_rate = len(oov_tokens) / len(test_vocab) if test_vocab else 0
        
        return oov_rate
    
    def calculate_semantic_similarity(self, original_tokens: List[str], processed_tokens: List[str]) -> float:
        """Расчет семантического сходства (упрощенная версия)"""
        # Используем Jaccard similarity как простую метрику
        original_set = set(original_tokens)
        processed_set = set(processed_tokens)
        
        intersection = len(original_set & processed_set)
        union = len(original_set | processed_set)
        
        jaccard_similarity = intersection / union if union > 0 else 0
        return jaccard_similarity
    
    def analyze_corpus(self, texts: List[str], test_size: float = 0.2) -> Dict[str, Any]:
        """Анализ корпуса текстов"""
        logger.info(f"Анализ корпуса из {len(texts)} текстов")
        
        # Разделение на train/test
        split_idx = int(len(texts) * (1 - test_size))
        train_texts = texts[:split_idx]
        test_texts = texts[split_idx:]
        
        # Объединение текстов
        train_text = ' '.join(train_texts)
        test_text = ' '.join(test_texts)
        
        # Анализ токенизации
        train_results = self.analyze_text(train_text)
        test_results = self.analyze_text(test_text)
        
        # Сравнительный анализ
        comparison_results = {}
        
        for method_name in train_results.keys():
            if method_name in test_results:
                train_tokens = train_results[method_name]['tokens']
                test_tokens = test_results[method_name]['tokens']
                
                # Расчет OOV
                oov_rate = self.calculate_oov_rate(train_tokens, test_tokens)
                
                # Семантическое сходство
                semantic_similarity = self.calculate_semantic_similarity(train_tokens, test_tokens)
                
                comparison_results[method_name] = {
                    'train_vocab_size': train_results[method_name]['vocabulary_size'],
                    'test_vocab_size': test_results[method_name]['vocabulary_size'],
                    'oov_rate': oov_rate,
                    'semantic_similarity': semantic_similarity,
                    'train_processing_time': train_results[method_name]['processing_time'],
                    'test_processing_time': test_results[method_name]['processing_time'],
                    'avg_token_length': train_results[method_name]['avg_token_length']
                }
        
        return {
            'train_results': train_results,
            'test_results': test_results,
            'comparison': comparison_results
        }
    
    def save_results(self, results: Dict[str, Any], filepath: str):
        """Сохранение результатов анализа"""
        # Определяем путь относительно текущего файла
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(filepath):
            filepath = os.path.join(current_dir, filepath)
        
        # Создание DataFrame для сравнения
        comparison_data = []
        
        for method_name, metrics in results['comparison'].items():
            row = {
                'method': method_name,
                'train_vocab_size': metrics['train_vocab_size'],
                'test_vocab_size': metrics['test_vocab_size'],
                'oov_rate': metrics['oov_rate'],
                'semantic_similarity': metrics['semantic_similarity'],
                'train_processing_time': metrics['train_processing_time'],
                'test_processing_time': metrics['test_processing_time'],
                'avg_token_length': metrics['avg_token_length']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('oov_rate')  # Сортировка по OOV rate
        
        # Сохранение в CSV
        csv_path = filepath.replace('.json', '.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Сохранение полных результатов в JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"Результаты сохранены в {filepath} и {csv_path}")
        return df

def main():
    """Пример использования TokenizationAnalyzer"""
    # Создание анализатора
    analyzer = TokenizationAnalyzer(language='russian')
    
    # Пример текстов
    sample_texts = [
        "Это первый пример текста для анализа токенизации.",
        "Второй текст содержит различные знаки препинания! И числа: 123, 456.",
        "Третий текст с сокращениями: т.е., и т.д., г. Москва.",
        "Четвертый текст с URL: https://example.com и email: test@example.com",
        "Пятый текст с датами: 15.12.2023, время 14:30, сумма 1,500,000 руб."
    ]
    
    print("Анализ токенизации и нормализации")
    print("=" * 50)
    
    # Анализ корпуса
    results = analyzer.analyze_corpus(sample_texts)
    
    # Сохранение результатов
    df = analyzer.save_results(results, "tokenization_metrics.json")
    
    # Вывод результатов
    print("\nСравнительная таблица методов токенизации:")
    print(df.to_string(index=False))
    
    # Анализ нормализации на примере одного текста
    print("\n" + "=" * 50)
    print("Анализ нормализации:")
    
    sample_text = "Это пример текста для анализа различных методов нормализации слов."
    tokens = analyzer.naive_tokenization(sample_text)
    
    normalization_results = analyzer.analyze_normalization(tokens)
    
    for method, result in normalization_results.items():
        if 'error' not in result:
            print(f"\n{method}:")
            print(f"  Токенов: {result['token_count']}")
            print(f"  Уникальных: {result['unique_tokens']}")
            print(f"  Время обработки: {result['processing_time']:.4f}с")
            print(f"  Коэффициент сжатия: {result['compression_ratio']:.2f}")
            print(f"  Токены: {result['tokens'][:10]}...")  # Первые 10 токенов

if __name__ == "__main__":
    main()

