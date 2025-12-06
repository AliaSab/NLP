# Быстрый старт

## 1. Установка зависимостей

```bash
cd NLP/Task3
pip install -r requirements.txt
```

## 2. Подготовка данных (если еще не выполнено)

```bash
python prepare_corpus.py
```

Это создаст размеченные данные в папке `corpus/`.

## 3. Запуск полного пайплайна

### Бинарная классификация:
```bash
python run_pipeline.py \
    --train corpus/binary_classification/train.jsonl \
    --val corpus/binary_classification/validation.jsonl \
    --test corpus/binary_classification/test.jsonl \
    --task binary \
    --output results/binary
```

### Многоклассовая классификация:
```bash
python run_pipeline.py \
    --train corpus/multiclass_classification/train.jsonl \
    --val corpus/multiclass_classification/validation.jsonl \
    --test corpus/multiclass_classification/test.jsonl \
    --task multiclass \
    --output results/multiclass
```

## 4. Запуск веб-интерфейса

```bash
streamlit run web_interface.py
```

Интерфейс откроется в браузере по адресу `http://localhost:8501`

## 5. Использование отдельных модулей

### Препроцессинг:
```python
from text_preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()
tokens = preprocessor.preprocess("Ваш текст здесь")
```

### Обучение классической модели:
```python
from classical_classifiers import ClassicalClassifier

classifier = ClassicalClassifier(task_type='binary')
classifier.train_logistic_regression(X_train, y_train)
```

### Интерпретация:
```python
from model_interpretation import FeatureImportanceAnalyzer

analyzer = FeatureImportanceAnalyzer(model, vectorizer)
importance = analyzer.get_top_features(n=20)
```

## Структура результатов

После выполнения пайплайна в папке `results/` будут:

- `classical_models_results.json` - метрики всех моделей
- `*_feature_importance.png` - важность признаков
- `*_confusion_matrix.png` - матрицы ошибок
- `model_comparison.png` - сравнение моделей

## Примечания

- Для работы с русским языком установите: `python -m spacy download ru_core_news_sm`
- Некоторые библиотеки (AutoSklearn, TPOT) опциональны
- Для трансформеров рекомендуется использовать GPU

















