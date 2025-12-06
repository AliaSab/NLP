# Лабораторная работа №3: Классификация текстов

Полная реализация всех этапов лабораторной работы по классификации текстов.

## Структура проекта

```
Task3/
├── prepare_corpus.py              # Этап 1: Подготовка корпуса
├── text_preprocessing.py           # Этап 2: Препроцессинг и векторизация
├── classical_classifiers.py        # Этап 3: Классические ML-модели
├── neural_classifiers.py           # Этап 4: Нейросетевые и трансформерные модели
├── imbalance_handling.py          # Этап 5: Борьба с дисбалансом классов
├── hyperparameter_tuning.py       # Этап 6: Настройка гиперпараметров
├── model_interpretation.py         # Этап 7: Интерпретация и визуализация
├── web_interface.py                # Этап 8: Веб-интерфейс
├── run_pipeline.py                 # Главный скрипт для запуска пайплайна
├── requirements.txt                # Зависимости
├── corpus/                         # Размеченные данные
│   ├── binary_classification/
│   ├── multiclass_classification/
│   └── multilabel_classification/
└── results/                        # Результаты экспериментов
```

## Этап 1: Подготовка экспериментального корпуса

### Описание
Разметка корпуса для трех типов задач классификации:
- **Бинарная классификация**: тональность (позитивная/негативная)
- **Многоклассовая классификация**: темы (политика, экономика, спорт, культура)
- **Многометочная классификация**: несколько тем одновременно

### Использование
```bash
cd NLP/Task3
python prepare_corpus.py
```

### Результаты
- Размеченные данные в формате JSONL
- Балансировка классов (дисбаланс ≤ 3:1)
- Разделение на train/validation/test (70/15/15) с стратификацией

## Этап 2: Подготовка данных для классификации

### Модуль: `text_preprocessing.py`

#### Функциональность:
- **Текстовый препроцессинг**:
  - Очистка от HTML-тегов, URL, специальных символов
  - Приведение к нижнему регистру
  - Обработка эмодзи
  - Токенизация с использованием spaCy
  - Лемматизация
  - Удаление стоп-слов

- **Векторизация**:
  - Bag of Words
  - TF-IDF
  - N-grams
  - Word2Vec, GloVe, FastText (из ЛР №2)
  - BERT эмбеддинги (RuBERT)

- **Мета-признаки**:
  - Статистические: длина текста, средняя длина слова, количество уникальных слов
  - Синтаксические: доля знаков препинания, заглавных букв, цифр
  - Лингвистические: сложность текста (Flesch–Kincaid)

## Этап 3: Классические методы классификации

### Модуль: `classical_classifiers.py`

#### Реализованные алгоритмы:

**Базовые алгоритмы:**
- Логистическая регрессия (L1/L2 регуляризация)
- SVM (линейное ядро)
- Случайный лес
- Градиентный бустинг (XGBoost, CatBoost, LightGBM)

**Ансамблирование:**
- Bagging
- Stacking/Blending (Voting Classifier)
- Hard Voting и Soft Voting

**AutoML:**
- AutoSklearn (опционально)
- TPOT (опционально)

## Этап 4: Нейросетевые и трансформерные модели

### Модуль: `neural_classifiers.py`

#### Реализованные архитектуры:

**Простые архитектуры:**
- Многослойный персептрон (MLP)

**CNN:**
- Kim CNN (1D-свертки для N-gram паттернов)

**RNN:**
- LSTM/GRU
- Двунаправленные LSTM/GRU (BiRNN)

**Гибридные:**
- CNN + LSTM/GRU

**Трансформеры:**
- Fine-tuning RuBERT (DeepPavlov/rubert-base-cased)
- Адаптация для классификации

## Этап 5: Борьба с дисбалансом классов

### Модуль: `imbalance_handling.py`

#### Реализованные методы:

**Взвешивание классов:**
- `class_weight` в scikit-learn
- Веса в функциях потерь для нейросетей

**Сэмплирование:**
- Random Under/Over-sampling
- SMOTE
- ADASYN

**Аугментация текстов:**
- Замена синонимами
- Вставка/удаление/замена слов
- Back Translation (опционально, требует загрузки модели)
- Easy Data Augmentation (EDA)

## Этап 6: Настройка и оценка моделей

### Модуль: `hyperparameter_tuning.py`

#### Реализованные методы:

**Кросс-валидация:**
- Stratified K-Fold
- Time Series Split
- Group K-Fold

**Подбор гиперпараметров:**
- Grid Search
- Random Search
- Bayesian Optimization (Optuna, Hyperopt)

**Регуляризация:**
- L1, L2 регуляризация
- Dropout
- Weight decay
- Early Stopping
- ReduceLROnPlateau

**Метрики:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Log Loss

## Этап 7: Интерпретация и визуализация

### Модуль: `model_interpretation.py`

#### Реализованные методы:

**Важность признаков:**
- Анализ весов для линейных моделей
- Важность признаков для tree-based моделей
- Топ-N значимых слов для TF-IDF

**SHAP/LIME:**
- Локальная интерпретация предсказаний
- Визуализация влияния слов
- Глобальное понимание модели

**Визуализация:**
- Матрица ошибок
- ROC-кривые
- Precision-Recall кривые
- Визуализация эмбеддингов (t-SNE, UMAP)
- Карты внимания для трансформеров

**Анализ ошибок:**
- Поиск примеров с ошибками
- Анализ паттернов ошибок

## Этап 8: Веб-интерфейс

### Модуль: `web_interface.py`

#### Функциональность:

**Интерактивная классификация:**
- Поле ввода для тестовых текстов
- Выбор типа задачи и модели
- Отображение результатов с вероятностями
- Сравнение предсказаний разных моделей

**Анализ и интерпретация:**
- Визуализация SHAP/LIME объяснений
- Отображение карты внимания
- Анализ ошибок на тестовой выборке

**Сравнение моделей:**
- Таблица сравнения метрик
- Визуализация ROC-кривых и PR-кривых
- Сводная таблица результатов

### Запуск веб-интерфейса:
```bash
streamlit run web_interface.py
```

## Запуск полного пайплайна

### Использование главного скрипта:

```bash
# Базовый запуск
python run_pipeline.py \
    --train corpus/binary_classification/train.jsonl \
    --val corpus/binary_classification/validation.jsonl \
    --test corpus/binary_classification/test.jsonl \
    --task binary \
    --output results/binary

# С использованием эмбеддингов
python run_pipeline.py \
    --train corpus/binary_classification/train.jsonl \
    --val corpus/binary_classification/validation.jsonl \
    --test corpus/binary_classification/test.jsonl \
    --task binary \
    --use-embeddings \
    --embedding-model ../Task2/word2vec_skip-gram_size200_win10_min5.model \
    --output results/binary
```

### Что делает пайплайн:

1. Загружает данные из JSONL файлов
2. Выполняет препроцессинг и векторизацию
3. Обучает классические ML-модели
4. Обучает нейросетевые модели (опционально)
5. Выполняет интерпретацию лучшей модели
6. Создает визуализации и сохраняет результаты

## Установка зависимостей

```bash
pip install -r requirements.txt

# Для русского языка spaCy
python -m spacy download ru_core_news_sm

# Для NLTK (если нужно)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Примеры использования

### Пример 1: Обучение классической модели

```python
from text_preprocessing import prepare_features, load_jsonl
from classical_classifiers import ClassicalClassifier

# Загрузка данных
train_data = load_jsonl('corpus/binary_classification/train.jsonl')
test_data = load_jsonl('corpus/binary_classification/test.jsonl')

# Подготовка признаков
X_train, y_train = prepare_features(train_data, ...)
X_test, y_test = prepare_features(test_data, ...)

# Обучение
classifier = ClassicalClassifier(task_type='binary')
classifier.train_logistic_regression(X_train, y_train)

# Оценка
results = classifier.evaluate(classifier.models['logistic_regression'], X_test, y_test)
```

### Пример 2: Интерпретация модели

```python
from model_interpretation import FeatureImportanceAnalyzer, VisualizationTools

# Анализ важности признаков
analyzer = FeatureImportanceAnalyzer(model, vectorizer)
importance = analyzer.get_top_features(n=20)

# Визуализация
visualizer = VisualizationTools()
visualizer.plot_feature_importance(importance, save_path='importance.png')
```

## Результаты

После выполнения пайплайна в директории `results/` будут сохранены:

- `classical_models_results.json` - метрики классических моделей
- `*_feature_importance.png` - визуализация важности признаков
- `*_confusion_matrix.png` - матрицы ошибок
- `model_comparison.png` - сравнение моделей
- `*_error_analysis.json` - анализ ошибок

## Важные замечания

⚠️ **Ограничения:**
- Исходный корпус содержит 2000 статей (меньше требуемых 10 000)
- Некоторые библиотеки (AutoSklearn, TPOT) требуют дополнительной настройки
- Для трансформеров требуется значительный объем памяти

💡 **Рекомендации:**
- Используйте GPU для обучения трансформеров
- Начните с классических моделей для быстрого прототипирования
- Используйте методы борьбы с дисбалансом для улучшения качества

### Back Translation (опционально)

Метод back translation поддерживает несколько источников перевода:
- **Hugging Face** (может быть заблокирован в некоторых регионах)
- **Google Translate** (через googletrans)
- **Deep Translator** (альтернативный сервис)

**Для России (если Hugging Face заблокирован):**

1. **Рекомендуется использовать EDA вместо back translation:**
   ```python
   method='eda'  # Не требует внешних сервисов!
   ```

2. **Или установить альтернативные библиотеки:**
   ```bash
   pip install deep-translator
   ```

3. **Или использовать Yandex Translate API** (если есть API ключ)

**Подробнее:** См. `ALTERNATIVE_AUGMENTATION.md`

## Лицензия

Этот проект создан в рамках учебной лабораторной работы.
