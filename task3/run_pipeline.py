"""
Главный скрипт для запуска полного пайплайна классификации текстов.
Объединяет все этапы лабораторной работы №3.
"""
import os
import nltk
nltk.download('punkt_tab')
import json
import numpy as np
import logging
from typing import Dict, List, Any
import os
from pathlib import Path
import pickle
import joblib

# Импорты наших модулей
from text_preprocessing import (
    TextPreprocessor, Vectorizer, FeatureExtractor,
    prepare_features, load_jsonl
)
from classical_classifiers import ClassicalClassifier
from neural_classifiers import NeuralClassifier
from imbalance_handling import ImbalanceHandler
from hyperparameter_tuning import HyperparameterTuner, ModelEvaluator
from model_interpretation import (
    FeatureImportanceAnalyzer, VisualizationTools, ErrorAnalysis
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ClassificationPipeline:
    """Главный класс для запуска пайплайна классификации."""
    
    def __init__(self, task_type: str = 'binary', output_dir: str = 'results'):
        self.task_type = task_type
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Инициализация компонентов
        self.preprocessor = TextPreprocessor()
        self.vectorizer = Vectorizer()
        self.feature_extractor = FeatureExtractor()
        self.imbalance_handler = ImbalanceHandler()
        self.visualizer = VisualizationTools()
        
        # Классификаторы
        self.classical_classifier = ClassicalClassifier(task_type=task_type)
        self.neural_classifier = NeuralClassifier(task_type=task_type, num_classes=2 if task_type == 'binary' else 4)
        
        # Оценщики
        self.evaluator = ModelEvaluator(task_type=task_type)
        self.tuner = HyperparameterTuner(task_type=task_type)
    
    def load_data(self, train_path: str, val_path: str, test_path: str) -> Dict[str, Any]:
        """Загрузка данных."""
        logger.info("Загрузка данных...")
        
        train_data = load_jsonl(train_path)
        val_data = load_jsonl(val_path)
        test_data = load_jsonl(test_path)
        
        logger.info(f"Загружено: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
        
        return {
            'train': train_data,
            'val': val_data,
            'test': test_data
        }
    
    def prepare_data(self, data: Dict[str, Any], use_tfidf: bool = True,
                    use_embeddings: bool = False, embedding_model_path: str = None) -> Dict[str, Any]:
        """Подготовка данных и извлечение признаков."""
        logger.info("Подготовка данных и извлечение признаков...")
        
        # Объединяем train и val для обучения векторизатора
        all_train_data = data['train'] + data['val']
        
        # Подготовка признаков для train (с обучением векторизатора)
        X_train, y_train = prepare_features(
            all_train_data, self.preprocessor, self.feature_extractor,
            self.vectorizer, use_tfidf=use_tfidf, use_embeddings=use_embeddings,
            embedding_model_path=embedding_model_path,
            fit_vectorizer=True  # Обучаем векторизатор на train данных
        )
        
        # Подготовка признаков для test (только трансформация, без обучения)
        X_test, y_test = prepare_features(
            data['test'], self.preprocessor, self.feature_extractor,
            self.vectorizer, use_tfidf=use_tfidf, use_embeddings=use_embeddings,
            embedding_model_path=embedding_model_path,
            fit_vectorizer=False  # НЕ переобучаем векторизатор!
        )
        
        # Проверка размерности
        if X_train.shape[1] != X_test.shape[1]:
            logger.error(f"Несоответствие размерности признаков: train={X_train.shape[1]}, test={X_test.shape[1]}")
            logger.info("Попытка исправить...")
            # Если размерности не совпадают, используем только TF-IDF
            logger.warning("Используется только TF-IDF для совместимости размерностей")
            # Пересоздаем признаки только с TF-IDF
            all_texts_train = [f"{item.get('title', '')} {item.get('text', '')}" for item in all_train_data]
            all_texts_test = [f"{item.get('title', '')} {item.get('text', '')}" for item in data['test']]
            
            self.vectorizer.fit_tfidf(all_texts_train)
            X_train = self.vectorizer.transform_tfidf(all_texts_train).toarray()
            X_test = self.vectorizer.transform_tfidf(all_texts_test).toarray()
            
            # Добавляем статистические признаки
            stat_train = []
            stat_test = []
            for text in all_texts_train:
                stat_feat = self.feature_extractor.extract_statistical_features(text)
                stat_train.append(list(stat_feat.values()))
            for text in all_texts_test:
                stat_feat = self.feature_extractor.extract_statistical_features(text)
                stat_test.append(list(stat_feat.values()))
            
            X_train = np.hstack([X_train, np.array(stat_train)])
            X_test = np.hstack([X_test, np.array(stat_test)])
            
            logger.info(f"Исправлено: train={X_train.shape[1]}, test={X_test.shape[1]}")
        
        # Кодирование меток
        from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
        
        if self.task_type == 'multilabel':
            # Для многометочной классификации используем MultiLabelBinarizer
            label_encoder = MultiLabelBinarizer()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            logger.info(f"Размерность признаков: {X_train.shape[1]}")
            logger.info(f"Классы (multilabel): {label_encoder.classes_}")
            logger.info(f"Форма меток train: {y_train_encoded.shape}, test: {y_test_encoded.shape}")
        else:
            # Для бинарной и многоклассовой классификации используем LabelEncoder
            label_encoder = LabelEncoder()
            y_train_encoded = label_encoder.fit_transform(y_train)
            y_test_encoded = label_encoder.transform(y_test)
            logger.info(f"Размерность признаков: {X_train.shape[1]}")
            logger.info(f"Классы: {label_encoder.classes_}")
        
        result = {
            'X_train': X_train,
            'y_train': y_train_encoded,
            'X_test': X_test,
            'y_test': y_test_encoded,
            'label_encoder': label_encoder,
            'texts_train': [f"{item.get('title', '')} {item.get('text', '')}" for item in all_train_data],
            'texts_test': [f"{item.get('title', '')} {item.get('text', '')}" for item in data['test']]
        }
        
        # Сохраняем label_encoder для использования в interpret_models
        self.label_encoder = label_encoder
        
        return result
    
    def train_classical_models(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Обучение классических моделей."""
        logger.info("Обучение классических моделей...")
        
        results = self.classical_classifier.train_all_models(X_train, y_train, X_test, y_test)
        
        # Сохранение результатов
        results_path = os.path.join(self.output_dir, 'classical_models_results.json')
        self.classical_classifier.save_results(results_path)
        
        # Сохранение моделей
        models_dir = os.path.join(self.output_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.classical_classifier.models.items():
            model_path = os.path.join(models_dir, f'{model_name}.pkl')
            try:
                joblib.dump(model, model_path)
                logger.info(f"Модель {model_name} сохранена в {model_path}")
            except Exception as e:
                logger.error(f"Ошибка сохранения модели {model_name}: {e}")
        
        return results
    
    def train_neural_models(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Обучение нейросетевых моделей."""
        logger.info("Обучение нейросетевых моделей...")
        
        results = {}
        
        # MLP
        try:
            mlp = self.neural_classifier.build_mlp(X_train.shape[1])
            if mlp:
                history = self.neural_classifier.train_keras_model(
                    mlp, X_train, y_train, X_val, y_val, epochs=10, model_name='mlp'
                )
                mlp_results = self.neural_classifier.evaluate_neural_model(mlp, X_test, y_test, 'mlp')
                results['mlp'] = mlp_results
        except Exception as e:
            logger.error(f"Ошибка обучения MLP: {e}")
        
        return results
    
    def interpret_models(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                        texts_test: List[str], model_name: str = 'model'):
        """Интерпретация моделей."""
        logger.info(f"Интерпретация модели: {model_name}")
        
        # Анализ важности признаков
        importance_analyzer = FeatureImportanceAnalyzer(model, self.vectorizer.tfidf_vectorizer)
        importance = importance_analyzer.get_top_features(n=20)
        
        # Визуализация важности признаков
        importance_path = os.path.join(self.output_dir, f'{model_name}_feature_importance.png')
        self.visualizer.plot_feature_importance(importance, save_path=importance_path)
        
        # Анализ ошибок
        error_analyzer = ErrorAnalysis(model, X_test, y_test, texts_test)
        errors = error_analyzer.find_errors(n_errors=10)
        error_patterns = error_analyzer.analyze_error_patterns()
        
        # Сохранение анализа ошибок
        error_path = os.path.join(self.output_dir, f'{model_name}_error_analysis.json')
        with open(error_path, 'w', encoding='utf-8') as f:
            json.dump(error_patterns, f, ensure_ascii=False, indent=2)
        
        # Визуализация матрицы ошибок
        y_pred = model.predict(X_test)
        cm_path = os.path.join(self.output_dir, f'{model_name}_confusion_matrix.png')
        
        # Для multilabel задач confusion matrix не применима напрямую
        if self.task_type != 'multilabel':
            # Получаем имена классов из label_encoder, если доступен
            class_names = None
            if hasattr(self, 'label_encoder') and self.label_encoder is not None:
                class_names = [str(cls) for cls in self.label_encoder.classes_]
            
            self.visualizer.plot_confusion_matrix(y_test, y_pred, class_names=class_names, save_path=cm_path)
        else:
            logger.info("Для multilabel задач confusion matrix не строится (используйте метрики по классам)")
    
    def run_full_pipeline(self, train_path: str, val_path: str, test_path: str,
                         use_tfidf: bool = True, use_embeddings: bool = False,
                         embedding_model_path: str = None):
        """Запуск полного пайплайна."""
        logger.info("="*60)
        logger.info("Запуск полного пайплайна классификации")
        logger.info("="*60)
        
        # 1. Загрузка данных
        data = self.load_data(train_path, val_path, test_path)
        
        # Логирование размеров выборок
        logger.info(f"Размеры выборок: train={len(data['train'])}, val={len(data['val'])}, test={len(data['test'])}")
        
        if len(data['test']) < 10:
            logger.warning(f"⚠️ Тестовая выборка очень мала ({len(data['test'])} образцов)! Результаты могут быть нерепрезентативными.")
        
        # 2. Подготовка данных
        prepared_data = self.prepare_data(
            data, use_tfidf=use_tfidf, use_embeddings=use_embeddings,
            embedding_model_path=embedding_model_path
        )
        
        logger.info(f"Размерности после подготовки: X_train={prepared_data['X_train'].shape}, X_test={prepared_data['X_test'].shape}")
        
        # 3. Обработка дисбаланса (опционально)
        # X_train_balanced, y_train_balanced = self.imbalance_handler.handle_imbalance(
        #     prepared_data['X_train'], prepared_data['y_train'],
        #     prepared_data['texts_train'], method='smote'
        # )
        
        # 4. Обучение классических моделей
        classical_results = self.train_classical_models(
            prepared_data['X_train'], prepared_data['y_train'],
            prepared_data['X_test'], prepared_data['y_test']
        )
        
        # 5. Обучение нейросетевых моделей (опционально)
        # neural_results = self.train_neural_models(...)
        
        # 6. Интерпретация лучшей модели
        best_model_name = max(classical_results.keys(), 
                            key=lambda x: classical_results[x].get('f1_score', 0))
        best_model = self.classical_classifier.models[best_model_name]
        
        self.interpret_models(
            best_model, prepared_data['X_test'], prepared_data['y_test'],
            prepared_data['texts_test'], best_model_name
        )
        
        # 7. Сравнение моделей
        comparison_path = os.path.join(self.output_dir, 'model_comparison.png')
        self.visualizer.plot_model_comparison(
            classical_results, metric='f1_score', save_path=comparison_path
        )
        
        # 8. Сохранение векторизатора и label_encoder для использования в веб-интерфейсе
        vectorizer_path = os.path.join(self.output_dir, 'vectorizer.pkl')
        try:
            joblib.dump(self.vectorizer, vectorizer_path)
            logger.info(f"Векторизатор сохранен в {vectorizer_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения векторизатора: {e}")
        
        if hasattr(self, 'label_encoder') and self.label_encoder is not None:
            label_encoder_path = os.path.join(self.output_dir, 'label_encoder.pkl')
            try:
                joblib.dump(self.label_encoder, label_encoder_path)
                logger.info(f"Label encoder сохранен в {label_encoder_path}")
            except Exception as e:
                logger.error(f"Ошибка сохранения label encoder: {e}")
        
        # Сохранение метаданных
        if hasattr(self, 'label_encoder') and self.label_encoder is not None:
            if self.task_type == 'multilabel':
                # Для multilabel classes_ - это массив классов
                num_classes = len(self.label_encoder.classes_)
                class_names = list(self.label_encoder.classes_)
            else:
                # Для binary/multiclass classes_ - это массив классов
                num_classes = len(self.label_encoder.classes_)
                class_names = list(self.label_encoder.classes_)
        else:
            num_classes = None
            class_names = None
        
        metadata = {
            'task_type': self.task_type,
            'feature_dim': prepared_data['X_train'].shape[1],
            'num_classes': num_classes,
            'class_names': class_names
        }
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info("="*60)
        logger.info("Пайплайн завершен!")
        logger.info(f"Результаты сохранены в: {self.output_dir}")
        logger.info("="*60)


def main():
    """Главная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Пайплайн классификации текстов')
    parser.add_argument('--task', type=str, default='binary',
                       choices=['binary', 'multiclass', 'multilabel'],
                       help='Тип задачи классификации')
    parser.add_argument('--train', type=str, required=True, help='Путь к train.jsonl')
    parser.add_argument('--val', type=str, required=True, help='Путь к validation.jsonl')
    parser.add_argument('--test', type=str, required=True, help='Путь к test.jsonl')
    parser.add_argument('--output', type=str, default='results', help='Директория для результатов')
    parser.add_argument('--use-embeddings', action='store_true', help='Использовать эмбеддинги')
    parser.add_argument('--embedding-model', type=str, help='Путь к модели эмбеддингов')
    
    args = parser.parse_args()
    
    # Создание пайплайна
    pipeline = ClassificationPipeline(task_type=args.task, output_dir=args.output)
    
    # Запуск пайплайна
    pipeline.run_full_pipeline(
        args.train, args.val, args.test,
        use_embeddings=args.use_embeddings,
        embedding_model_path=args.embedding_model
    )


if __name__ == '__main__':
    # Пример использования без аргументов командной строки
    if len(os.sys.argv) == 1:
        # Запуск для бинарной классификации
        pipeline = ClassificationPipeline(task_type='binary', output_dir='results/binary')
        
        train_path = 'corpus/binary_classification/train.jsonl'
        val_path = 'corpus/binary_classification/validation.jsonl'
        test_path = 'corpus/binary_classification/test.jsonl'
        
        if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
            pipeline.run_full_pipeline(train_path, val_path, test_path)
        else:
            logger.error("Файлы данных не найдены!")
            logger.info("Использование: python run_pipeline.py --train <path> --val <path> --test <path>")
    else:
        main()

