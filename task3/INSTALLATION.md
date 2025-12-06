# Установка библиотек

## Обязательные библиотеки

Эти библиотеки необходимы для базовой работы пайплайна:

```bash
pip install numpy pandas scikit-learn
```

## Рекомендуемые библиотеки

Для полной функциональности рекомендуется установить:

### NLP и векторизация:
```bash
pip install spacy nltk gensim
python -m spacy download ru_core_news_sm  # Для русского языка
```

### Классические ML модели:
```bash
pip install xgboost catboost lightgbm
```

### Нейросети:
```bash
pip install tensorflow torch transformers
```

### Обработка дисбаланса:
```bash
pip install imbalanced-learn nlpaug
```

### Настройка гиперпараметров:
```bash
pip install optuna hyperopt
```

### Интерпретация:
```bash
pip install shap lime
```

### Визуализация:
```bash
pip install matplotlib seaborn plotly umap-learn
```

### Веб-интерфейс:
```bash
pip install streamlit
```

## Опциональные библиотеки

Эти библиотеки не обязательны, но расширяют функциональность:

### AutoML (требуют дополнительной настройки):
```bash
# AutoSklearn - требует компилятор C++
pip install auto-sklearn

# TPOT
pip install tpot
```

## Быстрая установка всех рекомендуемых библиотек

```bash
pip install -r requirements.txt
```

## Минимальная установка для базовой работы

Если вы хотите использовать только классические ML-модели без нейросетей:

```bash
pip install numpy pandas scikit-learn xgboost catboost lightgbm imbalanced-learn matplotlib seaborn
```

## Проверка установки

После установки вы можете проверить доступность библиотек:

```python
# Проверка основных библиотек
import numpy as np
import pandas as pd
import sklearn
print("✓ Основные библиотеки установлены")

# Проверка NLP
try:
    import spacy
    print("✓ spaCy установлен")
except ImportError:
    print("✗ spaCy не установлен")

# Проверка ML моделей
try:
    import xgboost
    print("✓ XGBoost установлен")
except ImportError:
    print("✗ XGBoost не установлен")

# Проверка нейросетей
try:
    import tensorflow
    print("✓ TensorFlow установлен")
except ImportError:
    print("✗ TensorFlow не установлен")

try:
    import torch
    print("✓ PyTorch установлен")
except ImportError:
    print("✗ PyTorch не установлен")
```

## Примечания

- **TensorFlow/PyTorch**: Требуют значительный объем памяти. Для CPU-версий используйте:
  ```bash
  pip install tensorflow-cpu torch --index-url https://download.pytorch.org/whl/cpu
  ```

- **AutoSklearn**: Требует компилятор C++ (Visual Studio на Windows, gcc на Linux/Mac)

- **Transformers**: Для работы с русскими моделями может потребоваться дополнительное место на диске (модели весят несколько ГБ)

- **spaCy**: После установки необходимо загрузить языковую модель:
  ```bash
  python -m spacy download ru_core_news_sm
  ```

## Решение проблем

### Ошибка "ModuleNotFoundError"
Если вы видите ошибку о том, что модуль не найден, установите его:
```bash
pip install <название_модуля>
```

### Ошибка с torch.Tensor в аннотациях типов
Эта ошибка исправлена в коде - библиотеки теперь импортируются условно и не вызывают ошибок при отсутствии.

### Проблемы с AutoSklearn
AutoSklearn требует компилятор C++. Если у вас проблемы с установкой, просто пропустите эту библиотеку - она опциональна.

















