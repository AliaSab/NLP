import os
os.environ['TCL_LIBRARY'] = "C:/Program Files/Python313/tcl/tcl8.6"
os.environ['TK_LIBRARY'] = "C:/Program Files/Python313/tcl/tk8.6"
import json
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from collections import Counter
import re

# Импорт библиотек для подсловной токенизации
try:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
    from tokenizers.normalizers import NFD, Lowercase, Sequence
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False
    print("tokenizers не установлен. Установите: pip install tokenizers")

try:
    import sentencepiece as sp
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    print("sentencepiece не установлен. Установите: pip install sentencepiece")

logger = logging.getLogger(__name__)

class SubwordModelTrainer:
    """Тренер для подсловных моделей токенизации"""
    
    def __init__(self, language: str = 'russian'):
        self.language = language
        self.models = {}
        self.results = {}
    
    def prepare_corpus(self, texts: List[str], output_file: str = "corpus.txt"):
        """Подготовка корпуса для обучения"""
        logger.info(f"Подготовка корпуса из {len(texts)} текстов")
        
        # Объединение текстов
        corpus_text = '\n'.join(texts)
        
        # Сохранение корпуса
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(corpus_text)
        
        logger.info(f"Корпус сохранен в {output_file}")
        return output_file
    
    def train_bpe_model(self, corpus_file: str, vocab_size: int = 8000, 
                       min_frequency: int = 2) -> Dict[str, Any]:
        """Обучение BPE модели"""
        if not TOKENIZERS_AVAILABLE:
            logger.error("tokenizers не установлен")
            return {}
        
        logger.info(f"Обучение BPE модели с vocab_size={vocab_size}")
        
        try:
            # Создание токенизатора
            tokenizer = Tokenizer(models.BPE())
            tokenizer.normalizer = Sequence([NFD(), Lowercase()])
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            
            # Тренер
            trainer = trainers.BpeTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"]
            )
            
            # Обучение
            start_time = time.time()
            tokenizer.train([corpus_file], trainer)
            training_time = time.time() - start_time
            
            # Сохранение модели
            model_path = f"bpe_model_{vocab_size}.json"
            tokenizer.save(model_path)
            
            # Тестирование
            test_text = "Это пример текста для тестирования BPE модели."
            encoding = tokenizer.encode(test_text)
            
            results = {
                'model_type': 'BPE',
                'vocab_size': vocab_size,
                'min_frequency': min_frequency,
                'training_time': training_time,
                'model_path': model_path,
                'test_tokens': encoding.tokens,
                'test_token_count': len(encoding.tokens),
                'tokenizer': tokenizer
            }
            
            self.models['bpe'] = tokenizer
            logger.info(f"BPE модель обучена за {training_time:.2f}с")
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка обучения BPE модели: {e}")
            return {}
    
    def train_wordpiece_model(self, corpus_file: str, vocab_size: int = 8000,
                            min_frequency: int = 2) -> Dict[str, Any]:
        """Обучение WordPiece модели"""
        if not TOKENIZERS_AVAILABLE:
            logger.error("tokenizers не установлен")
            return {}
        
        logger.info(f"Обучение WordPiece модели с vocab_size={vocab_size}")
        
        try:
            # Создание токенизатора
            tokenizer = Tokenizer(models.WordPiece())
            tokenizer.normalizer = Sequence([NFD(), Lowercase()])
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            
            # Тренер
            trainer = trainers.WordPieceTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"]
            )
            
            # Обучение
            start_time = time.time()
            tokenizer.train([corpus_file], trainer)
            training_time = time.time() - start_time
            
            # Сохранение модели
            model_path = f"wordpiece_model_{vocab_size}.json"
            tokenizer.save(model_path)
            
            # Тестирование
            test_text = "Это пример текста для тестирования WordPiece модели."
            encoding = tokenizer.encode(test_text)
            
            results = {
                'model_type': 'WordPiece',
                'vocab_size': vocab_size,
                'min_frequency': min_frequency,
                'training_time': training_time,
                'model_path': model_path,
                'test_tokens': encoding.tokens,
                'test_token_count': len(encoding.tokens),
                'tokenizer': tokenizer
            }
            
            self.models['wordpiece'] = tokenizer
            logger.info(f"WordPiece модель обучена за {training_time:.2f}с")
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка обучения WordPiece модели: {e}")
            return {}
    
    def train_unigram_model(self, corpus_file: str, vocab_size: int = 8000,
                          min_frequency: int = 2) -> Dict[str, Any]:
        """Обучение Unigram модели"""
        if not TOKENIZERS_AVAILABLE:
            logger.error("tokenizers не установлен")
            return {}
        
        logger.info(f"Обучение Unigram модели с vocab_size={vocab_size}")
        
        try:
            # Создание токенизатора
            tokenizer = Tokenizer(models.Unigram())
            tokenizer.normalizer = Sequence([NFD(), Lowercase()])
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            
            # Тренер
            trainer = trainers.UnigramTrainer(
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"]
            )
            
            # Обучение
            start_time = time.time()
            tokenizer.train([corpus_file], trainer)
            training_time = time.time() - start_time
            
            # Сохранение модели
            model_path = f"unigram_model_{vocab_size}.json"
            tokenizer.save(model_path)
            
            # Тестирование
            test_text = "Это пример текста для тестирования Unigram модели."
            encoding = tokenizer.encode(test_text)
            
            results = {
                'model_type': 'Unigram',
                'vocab_size': vocab_size,
                'min_frequency': min_frequency,
                'training_time': training_time,
                'model_path': model_path,
                'test_tokens': encoding.tokens,
                'test_token_count': len(encoding.tokens),
                'tokenizer': tokenizer
            }
            
            self.models['unigram'] = tokenizer
            logger.info(f"Unigram модель обучена за {training_time:.2f}с")
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка обучения Unigram модели: {e}")
            return {}
    
    def train_sentencepiece_model(self, corpus_file: str, vocab_size: int = 8000,
                                model_type: str = 'unigram') -> Dict[str, Any]:
        """Обучение SentencePiece модели"""
        if not SENTENCEPIECE_AVAILABLE:
            logger.error("sentencepiece не установлен")
            return {}
        
        logger.info(f"Обучение SentencePiece {model_type} модели с vocab_size={vocab_size}")
        
        try:
            # Параметры обучения
            model_prefix = f"sp_{model_type}_{vocab_size}"
            
            sp.SentencePieceTrainer.train(
                input=corpus_file,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=0.9995,
                num_threads=4,
                input_sentence_size=1000000,
                shuffle_input_sentence=True,
                normalization_rule_name='nmt_nfkc_cf'
            )
            
            # Загрузка модели
            sp_model = sp.SentencePieceProcessor()
            sp_model.load(f"{model_prefix}.model")
            
            # Тестирование
            test_text = "Это пример текста для тестирования SentencePiece модели."
            tokens = sp_model.encode(test_text, out_type=str)
            
            results = {
                'model_type': f'SentencePiece_{model_type}',
                'vocab_size': vocab_size,
                'training_time': 0,  # SentencePiece не возвращает время обучения
                'model_path': f"{model_prefix}.model",
                'test_tokens': tokens,
                'test_token_count': len(tokens),
                'tokenizer': sp_model
            }
            
            self.models[f'sentencepiece_{model_type}'] = sp_model
            logger.info(f"SentencePiece {model_type} модель обучена")
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка обучения SentencePiece модели: {e}")
            return {}
    
    def evaluate_model(self, model_name: str, test_texts: List[str]) -> Dict[str, Any]:
        """Оценка модели на тестовых данных"""
        if model_name not in self.models:
            logger.error(f"Модель {model_name} не найдена")
            return {}
        
        model = self.models[model_name]
        
        # Метрики
        total_tokens = 0
        total_words = 0
        fragmented_words = 0
        processing_times = []
        
        for text in test_texts:
            start_time = time.time()
            
            # Определяем тип модели по имени
            if 'sentencepiece' in model_name.lower() or 'sp_' in model_name.lower():
                # SentencePiece модели
                tokens = model.encode(text, out_type=str)
            else:
                # Tokenizers модели (BPE, WordPiece, Unigram)
                encoding = model.encode(text)
                tokens = encoding.tokens
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            # Подсчет метрик
            words = text.split()
            total_words += len(words)
            total_tokens += len(tokens)
            
            # Подсчет фрагментированных слов
            for word in words:
                word_tokens = [token for token in tokens if word.lower() in token.lower()]
                if len(word_tokens) > 1:
                    fragmented_words += 1
        
        # Расчет метрик
        fragmentation_rate = fragmented_words / total_words if total_words > 0 else 0
        compression_ratio = total_words / total_tokens if total_tokens > 0 else 1
        avg_processing_time = np.mean(processing_times)
        
        results = {
            'model_name': model_name,
            'total_words': total_words,
            'total_tokens': total_tokens,
            'fragmented_words': fragmented_words,
            'fragmentation_rate': fragmentation_rate,
            'compression_ratio': compression_ratio,
            'avg_processing_time': avg_processing_time,
            'tokens_per_second': total_tokens / sum(processing_times) if sum(processing_times) > 0 else 0
        }
        
        return results
    
    def train_all_models(self, texts: List[str], vocab_sizes: List[int] = [8000, 16000, 32000]) -> Dict[str, Any]:
        """Обучение всех моделей с разными размерами словаря"""
        logger.info("Начало обучения всех подсловных моделей")
        
        # Подготовка корпуса
        corpus_file = self.prepare_corpus(texts)
        
        all_results = {}
        
        for vocab_size in vocab_sizes:
            logger.info(f"Обучение моделей с vocab_size={vocab_size}")
            
            # BPE
            bpe_results = self.train_bpe_model(corpus_file, vocab_size)
            if bpe_results:
                all_results[f'bpe_{vocab_size}'] = bpe_results
            
            # WordPiece
            wp_results = self.train_wordpiece_model(corpus_file, vocab_size)
            if wp_results:
                all_results[f'wordpiece_{vocab_size}'] = wp_results
            
            # Unigram
            unigram_results = self.train_unigram_model(corpus_file, vocab_size)
            if unigram_results:
                all_results[f'unigram_{vocab_size}'] = unigram_results
            
            # SentencePiece Unigram
            sp_unigram_results = self.train_sentencepiece_model(corpus_file, vocab_size, 'unigram')
            if sp_unigram_results:
                all_results[f'sp_unigram_{vocab_size}'] = sp_unigram_results
            
            # SentencePiece BPE
            sp_bpe_results = self.train_sentencepiece_model(corpus_file, vocab_size, 'bpe')
            if sp_bpe_results:
                all_results[f'sp_bpe_{vocab_size}'] = sp_bpe_results
        
        # Оценка всех моделей
        test_texts = texts[:100]  # Используем первые 100 текстов для тестирования
        evaluation_results = {}
        
        for model_name in self.models.keys():
            eval_results = self.evaluate_model(model_name, test_texts)
            if eval_results:
                evaluation_results[model_name] = eval_results
        
        # Сохранение результатов
        results = {
            'training_results': all_results,
            'evaluation_results': evaluation_results,
            'vocab_sizes': vocab_sizes,
            'corpus_size': len(texts)
        }
        
        # Сохранение в файл
        current_dir = os.path.dirname(os.path.abspath(__file__))
        results_path = os.path.join(current_dir, 'subword_models_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        # Создание сравнительной таблицы
        self.create_comparison_table(evaluation_results)
        
        logger.info("Обучение всех моделей завершено")
        return results
    
    def create_comparison_table(self, evaluation_results: Dict[str, Any]) -> pd.DataFrame:
        """Создание сравнительной таблицы моделей"""
        comparison_data = []
        
        for model_name, metrics in evaluation_results.items():
            row = {
                'model': model_name,
                'fragmentation_rate': metrics['fragmentation_rate'],
                'compression_ratio': metrics['compression_ratio'],
                'avg_processing_time': metrics['avg_processing_time'],
                'tokens_per_second': metrics['tokens_per_second'],
                'total_tokens': metrics['total_tokens'],
                'total_words': metrics['total_words']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('fragmentation_rate')
        
        # Сохранение таблицы
        current_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(current_dir, 'subword_models_comparison.csv')
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        logger.info("Сравнительная таблица сохранена в subword_models_comparison.csv")
        return df

def main():
    """Пример использования SubwordModelTrainer"""
    # Создание тренера
    trainer = SubwordModelTrainer(language='russian')
    
    # Пример текстов
    sample_texts = [
        "Это первый пример текста для обучения подсловных моделей.",
        "Второй текст содержит различные слова и фразы.",
        "Третий текст с техническими терминами и сокращениями.",
        "Четвертый текст на русском языке с пунктуацией!",
        "Пятый текст для тестирования эффективности моделей."
    ] * 20  # Дублируем для увеличения корпуса
    
    print("Обучение подсловных моделей токенизации")
    print("=" * 50)
    
    # Обучение всех моделей
    results = trainer.train_all_models(sample_texts, vocab_sizes=[8000, 16000])
    
    # Вывод результатов
    if 'evaluation_results' in results:
        print("\nРезультаты оценки моделей:")
        for model_name, metrics in results['evaluation_results'].items():
            print(f"\n{model_name}:")
            print(f"  Коэффициент фрагментации: {metrics['fragmentation_rate']:.4f}")
            print(f"  Коэффициент сжатия: {metrics['compression_ratio']:.4f}")
            print(f"  Среднее время обработки: {metrics['avg_processing_time']:.4f}с")
            print(f"  Токенов в секунду: {metrics['tokens_per_second']:.2f}")

if __name__ == "__main__":
    main()

