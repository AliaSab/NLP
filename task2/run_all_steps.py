"""
Главный файл для запуска всех этапов анализа векторных представлений текста.

Этапы:
1. Классическая векторизация (One-Hot, BoW, TF-IDF)
2. Снижение размерности и тематическое моделирование (SVD, LSA)
3. Сравнительный анализ классических методов
4. Обучение моделей распределённых представлений (Word2Vec, FastText, Doc2Vec)
5. Эксперименты с векторной арифметикой
6. Веб-интерфейс для анализа
"""
import os
import sys
import json
import time
import logging
from typing import Dict, Any
import pandas as pd

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Импорт модулей
from classical_vectorizers import ClassicalVectorizer
from dimensionality_reduction import DimensionalityReducer
from distributed_representations import DistributedRepresentationsTrainer
from vector_arithmetic import VectorArithmeticAnalyzer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vector_analysis.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VectorAnalysisPipeline:
    """Класс для управления пайплайном анализа векторных представлений."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.results = {}
        
    def run_step2_classical_vectorization(self) -> Dict[str, Any]:
        """Этап 2: Классическая векторизация."""
        logger.info("=== ЭТАП 2: КЛАССИЧЕСКАЯ ВЕКТОРИЗАЦИЯ ===")
        
        try:
            vectorizer = ClassicalVectorizer()
            texts = vectorizer.load_data(self.data_path)
            
            if not texts:
                logger.error("Не удалось загрузить данные")
                return {}
            
            # Ограничиваем количество текстов для демонстрации
            texts = texts[:1000]
            logger.info(f"Обработка {len(texts)} текстов")
            
            # Запуск всех экспериментов
            results = vectorizer.run_all_experiments(texts)
            
            # Создание сводной таблицы
            summary_df = vectorizer.create_summary_table()
            summary_df.to_csv("vectorization_metrics.csv", index=False, encoding='utf-8')
            
            # Сохранение результатов
            vectorizer.save_results("classical_vectorization_results.json")
            
            logger.info("Этап 2 завершен")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в этапе 2: {e}")
            return {}
    
    def run_step3_dimensionality_reduction(self) -> Dict[str, Any]:
        """Этап 3: Снижение размерности и тематическое моделирование."""
        logger.info("=== ЭТАП 3: СНИЖЕНИЕ РАЗМЕРНОСТИ ===")
        
        try:
            # Загрузка TF-IDF матрицы из предыдущего этапа
            vectorizer = ClassicalVectorizer()
            texts = vectorizer.load_data(self.data_path)
            
            if not texts:
                logger.error("Не удалось загрузить данные")
                return {}
            
            texts = texts[:1000]
            
            # Создание TF-IDF матрицы
            tfidf_result = vectorizer.tfidf_vectorization(texts, ngram_range=(1, 2))
            matrix = tfidf_result['matrix']
            feature_names = tfidf_result['feature_names']
            
            # Запуск анализа снижения размерности
            reducer = DimensionalityReducer()
            results = reducer.run_comprehensive_analysis(matrix, feature_names)
            
            logger.info("Этап 3 завершен")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в этапе 3: {e}")
            return {}
    
    def run_step4_comparative_analysis(self) -> Dict[str, Any]:
        """Этап 4: Сравнительный анализ классических методов."""
        logger.info("=== ЭТАП 4: СРАВНИТЕЛЬНЫЙ АНАЛИЗ ===")
        
        try:
            # Загрузка результатов предыдущих этапов
            try:
                with open("classical_vectorization_results.json", 'r', encoding='utf-8') as f:
                    classical_results = json.load(f)
            except FileNotFoundError:
                logger.error("Результаты классической векторизации не найдены")
                return {}
            
            # Создание сравнительного анализа
            comparison_results = self.create_comparative_analysis(classical_results)
            
            # Сохранение результатов
            with open("comparative_analysis_results.json", 'w', encoding='utf-8') as f:
                json.dump(comparison_results, f, ensure_ascii=False, indent=2)
            
            logger.info("Этап 4 завершен")
            return comparison_results
            
        except Exception as e:
            logger.error(f"Ошибка в этапе 4: {e}")
            return {}
    
    def create_comparative_analysis(self, classical_results: Dict[str, Any]) -> Dict[str, Any]:
        """Создание сравнительного анализа методов."""
        comparison = {
            'summary': {},
            'recommendations': {},
            'performance_metrics': {}
        }
        
        # Анализ по категориям методов
        for category, methods in classical_results.items():
            if category == 'ngram_comparison':
                continue
            
            # Проверяем, что methods является словарем
            if not isinstance(methods, dict):
                logger.warning(f"Пропускаем категорию {category}: неверный формат данных")
                continue
                
            category_summary = {
                'best_method': None,
                'best_score': 0,
                'methods_count': len(methods)
            }
            
            for method_name, method_data in methods.items():
                # Проверяем, что method_data является словарем
                if not isinstance(method_data, dict):
                    logger.warning(f"Пропускаем метод {method_name}: неверный формат данных")
                    continue
                
                analysis = method_data.get('analysis', {})
                
                # Комплексная оценка (чем меньше разреженность и время обработки, тем лучше)
                sparsity = analysis.get('sparsity', 0.5)  # По умолчанию 50% разреженность
                processing_time = analysis.get('processing_time', 1.0)  # По умолчанию 1 секунда
                score = (1 - sparsity) / (1 + processing_time)
                
                if score > category_summary['best_score']:
                    category_summary['best_method'] = method_name
                    category_summary['best_score'] = score
            
            comparison['summary'][category] = category_summary
        
        # Рекомендации
        comparison['recommendations'] = {
            'for_small_datasets': 'TF-IDF с n-граммами (1,2)',
            'for_large_datasets': 'TF-IDF с ограниченным количеством признаков',
            'for_speed': 'One-Hot Encoding',
            'for_quality': 'TF-IDF с настройкой параметров'
        }
        
        return comparison
    
    def run_step5_distributed_representations(self) -> Dict[str, Any]:
        """Этап 5: Обучение моделей распределённых представлений."""
        logger.info("=== ЭТАП 5: РАСПРЕДЕЛЁННЫЕ ПРЕДСТАВЛЕНИЯ ===")
        
        try:
            trainer = DistributedRepresentationsTrainer()
            results = trainer.run_training_pipeline(self.data_path)
            
            logger.info("Этап 5 завершен")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в этапе 5: {e}")
            return {}
    
    def run_step6_vector_arithmetic(self) -> Dict[str, Any]:
        """Этап 6: Эксперименты с векторной арифметикой."""
        logger.info("=== ЭТАП 6: ВЕКТОРНАЯ АРИФМЕТИКА ===")
        
        try:
            # Поиск обученных моделей
            model_files = []
            for file in os.listdir('.'):
                if file.endswith('.model'):
                    model_files.append(file)
            
            if not model_files:
                logger.warning("Обученные модели не найдены. Сначала запустите этап 5.")
                return {}
            
            # Анализ первой найденной модели
            model_path = model_files[0]
            logger.info(f"Анализ модели: {model_path}")
            
            analyzer = VectorArithmeticAnalyzer()
            results = analyzer.run_comprehensive_analysis(model_path, 'word2vec')
            
            logger.info("Этап 6 завершен")
            return results
            
        except Exception as e:
            logger.error(f"Ошибка в этапе 6: {e}")
            return {}
    
    def run_step7_web_interface(self) -> None:
        """Этап 7: Запуск веб-интерфейса."""
        logger.info("=== ЭТАП 7: ВЕБ-ИНТЕРФЕЙС ===")
        
        logger.info("Для запуска веб-интерфейса выполните команду:")
        logger.info("streamlit run web_interface.py")
        
        logger.info("Этап 7 - инструкции предоставлены")
    
    def run_all_steps(self) -> Dict[str, Any]:
        """Запуск всех этапов анализа."""
        logger.info("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА АНАЛИЗА ВЕКТОРНЫХ ПРЕДСТАВЛЕНИЙ")
        
        start_time = time.time()
        
        try:
            # Этап 2: Классическая векторизация
            logger.info("Запуск этапа 2...")
            self.results['step2'] = self.run_step2_classical_vectorization()
            
            # Этап 3: Снижение размерности
            logger.info("Запуск этапа 3...")
            self.results['step3'] = self.run_step3_dimensionality_reduction()
            
            # Этап 4: Сравнительный анализ
            logger.info("Запуск этапа 4...")
            self.results['step4'] = self.run_step4_comparative_analysis()
            
            # Этап 5: Распределённые представления
            logger.info("Запуск этапа 5...")
            self.results['step5'] = self.run_step5_distributed_representations()
            
            # Этап 6: Векторная арифметика
            logger.info("Запуск этапа 6...")
            self.results['step6'] = self.run_step6_vector_arithmetic()
            
            # Этап 7: Веб-интерфейс
            logger.info("Запуск этапа 7...")
            self.run_step7_web_interface()
            
            total_time = time.time() - start_time
            
            # Создание итогового отчёта
            self.create_final_report(total_time)
            
            logger.info(f"ПАЙПЛАЙН ЗАВЕРШЕН ЗА {total_time:.2f} СЕКУНД")
            
        except Exception as e:
            logger.error(f"Критическая ошибка в пайплайне: {e}")
            # Создаем отчёт даже при ошибке
            try:
                self.create_final_report(time.time() - start_time)
            except:
                pass
            raise
        
        return self.results
    
    def create_final_report(self, total_time: float) -> None:
        """Создание итогового отчёта."""
        logger.info("Создание итогового отчёта")
        
        report = {
            'pipeline_summary': {
                'total_execution_time': total_time,
                'steps_completed': len(self.results),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'step_summaries': {}
        }
        
        # Сводка по каждому этапу
        for step, results in self.results.items():
            if step == 'step2':
                report['step_summaries'][step] = {
                    'name': 'Классическая векторизация',
                    'methods_tested': len(results) if isinstance(results, dict) else 0,
                    'status': 'completed'
                }
            elif step == 'step3':
                report['step_summaries'][step] = {
                    'name': 'Снижение размерности',
                    'status': 'completed'
                }
            elif step == 'step4':
                report['step_summaries'][step] = {
                    'name': 'Сравнительный анализ',
                    'status': 'completed'
                }
            elif step == 'step5':
                report['step_summaries'][step] = {
                    'name': 'Распределённые представления',
                    'word_models': len(results.get('word_models', {})),
                    'doc_models': len(results.get('doc_models', {})),
                    'status': 'completed'
                }
            elif step == 'step6':
                report['step_summaries'][step] = {
                    'name': 'Векторная арифметика',
                    'status': 'completed'
                }
        
        # Сохранение отчёта
        with open('pipeline_final_report.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info("Итоговый отчёт сохранен в pipeline_final_report.json")


def main():
    """Основная функция."""
    # Путь к данным
    data_path = "C:/Users/alia2/OneDrive/Desktop/NLP_ALYA/kommersant_articles_processed.jsonl"
    
    if not os.path.exists(data_path):
        logger.error(f"Файл данных не найден: {data_path}")
        logger.info("Убедитесь, что файл kommersant_articles_processed.jsonl находится в папке Task1")
        return
    
    # Создание и запуск пайплайна
    pipeline = VectorAnalysisPipeline(data_path)
    
    try:
        results = pipeline.run_all_steps()
        
        print("\n" + "="*60)
        print("ПАЙПЛАЙН АНАЛИЗА ВЕКТОРНЫХ ПРЕДСТАВЛЕНИЙ ЗАВЕРШЕН")
        print("="*60)
        
        print("\nСозданные файлы:")
        print("• vectorization_metrics.csv - метрики векторизации")
        print("• classical_vectorization_results.json - результаты классической векторизации")
        print("• dimensionality_reduction_results.json - результаты снижения размерности")
        print("• variance_analysis.csv - анализ объясненной дисперсии")
        print("• distributed_representations_results.json - результаты обучения моделей")
        print("• word_models_summary.csv - сводка по word моделям")
        print("• doc_models_summary.csv - сводка по doc моделям")
        print("• vector_arithmetic_results.json - результаты векторной арифметики")
        print("• word_analogies_results.csv - результаты аналогий")
        print("• pipeline_final_report.json - итоговый отчёт")
        
        print("\nДля запуска веб-интерфейса выполните:")
        print("streamlit run web_interface.py")
        
    except Exception as e:
        logger.error(f"Ошибка выполнения пайплайна: {e}")
        print(f"\nОшибка: {e}")
        print("Проверьте логи в файле vector_analysis.log")


if __name__ == "__main__":
    main()
