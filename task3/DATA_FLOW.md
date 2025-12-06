# Поток данных в лабораторной работе №3

## Использование исходного файла `kommersant_articles_processed.jsonl`

### Где используется:

Файл `kommersant_articles_processed.jsonl` из `NLP/Task1/` используется **только на Этапе 1** в скрипте `prepare_corpus.py`:

```python
# prepare_corpus.py, строка 423
input_file = '../Task1/kommersant_articles_processed.jsonl'
```

### Что происходит:

1. **Этап 1** (`prepare_corpus.py`):
   - Загружает исходный файл `kommersant_articles_processed.jsonl`
   - Выполняет автоматическую разметку данных:
     - Определяет тональность (positive/negative)
     - Определяет категории (politics/economy/sport/culture)
     - Создает многометочную разметку
   - Балансирует классы
   - Разделяет на train/validation/test (70/15/15)
   - Сохраняет размеченные данные в `corpus/`

2. **Последующие этапы** (2-8):
   - Используют уже размеченные данные из папки `corpus/`:
     - `corpus/binary_classification/train.jsonl`
     - `corpus/binary_classification/validation.jsonl`
     - `corpus/binary_classification/test.jsonl`
     - И аналогично для `multiclass_classification/` и `multilabel_classification/`

### Поток данных:

```
kommersant_articles_processed.jsonl (Task1)
    ↓
prepare_corpus.py (Этап 1)
    ↓
corpus/
    ├── binary_classification/
    │   ├── train.jsonl
    │   ├── validation.jsonl
    │   └── test.jsonl
    ├── multiclass_classification/
    │   └── ...
    └── multilabel_classification/
        └── ...
    ↓
run_pipeline.py (Этапы 2-8)
    ├── text_preprocessing.py
    ├── classical_classifiers.py
    ├── neural_classifiers.py
    └── ...
```

### Важно:

- Исходный файл `kommersant_articles_processed.jsonl` используется **только один раз** при подготовке корпуса
- Все последующие этапы работают с размеченными данными из `corpus/`
- Если нужно пересоздать корпус, запустите `prepare_corpus.py` снова

### Пример использования:

```bash
# Шаг 1: Подготовка корпуса (использует kommersant_articles_processed.jsonl)
python prepare_corpus.py

# Шаг 2: Запуск пайплайна (использует данные из corpus/)
python run_pipeline.py \
    --train corpus/binary_classification/train.jsonl \
    --val corpus/binary_classification/validation.jsonl \
    --test corpus/binary_classification/test.jsonl
```

















