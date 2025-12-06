# Альтернативные методы аугментации (без Hugging Face)

Если Hugging Face заблокирован в вашем регионе, вы можете использовать альтернативные методы аугментации данных.

## Рекомендуемые методы (не требуют внешних сервисов)

### 1. Easy Data Augmentation (EDA) - **РЕКОМЕНДУЕТСЯ**

Самый простой и эффективный метод, не требующий внешних сервисов:

```python
from imbalance_handling import ImbalanceHandler

handler = ImbalanceHandler(language='ru')
augmented_texts, augmented_labels = handler.augmentation.augment_texts(
    texts, labels, target_class='positive',
    num_augmentations=2,
    method='eda'  # Используйте EDA вместо back_translation
)
```

**Преимущества:**
- Не требует интернет-соединения
- Работает полностью локально
- Быстрый
- Не требует установки дополнительных библиотек

### 2. Замена синонимами

```python
augmented_texts, augmented_labels = handler.augmentation.augment_texts(
    texts, labels, target_class='positive',
    num_augmentations=2,
    method='synonym'
)
```

## Альтернативные методы перевода (если нужен back translation)

### Вариант 1: Google Translate (googletrans)

```bash
pip install googletrans==4.0.0rc1
```

**Примечание:** Может быть нестабильным из-за изменений в API Google.

### Вариант 2: Deep Translator

```bash
pip install deep-translator
```

**Преимущества:**
- Более стабильный, чем googletrans
- Поддерживает множество языков
- Может работать через прокси

### Вариант 3: Yandex Translate API (для России)

Если у вас есть API ключ Yandex:

```python
from deep_translator import YandexTranslator

translator = YandexTranslator(api_key='YOUR_API_KEY', source='ru', target='en')
translated = translator.translate(text)
```

## Использование без back translation

В большинстве случаев back translation не обязателен. Рекомендуется использовать:

1. **EDA (Easy Data Augmentation)** - основной метод
2. **SMOTE/ADASYN** - для числовых признаков
3. **Random Over/Under sampling** - простой и эффективный

### Пример полного пайплайна без back translation:

```python
from imbalance_handling import ImbalanceHandler

handler = ImbalanceHandler(language='ru')

# Используем SMOTE для числовых признаков
X_balanced, y_balanced = handler.handle_imbalance(
    X, y, texts,
    method='smote'  # или 'adasyn', 'oversample'
)

# Или используем EDA для текстовой аугментации
augmented_texts, augmented_labels = handler.augmentation.augment_texts(
    texts, labels, target_class='minority_class',
    num_augmentations=3,
    method='eda'  # Не требует внешних сервисов!
)
```

## Рекомендации для России

1. **Используйте EDA** - самый надежный метод, работает без интернета
2. **Используйте SMOTE/ADASYN** - для векторизованных признаков
3. **Избегайте back translation** - если Hugging Face и Google заблокированы
4. **Рассмотрите Yandex Translate API** - если нужен перевод и есть API ключ

## Установка альтернативных библиотек

```bash
# Для EDA (уже включено в requirements.txt)
# Ничего дополнительного не требуется

# Для Google Translate (опционально)
pip install googletrans==4.0.0rc1

# Для Deep Translator (опционально)
pip install deep-translator

# Для Yandex (если есть API ключ)
pip install deep-translator  # уже включает Yandex
```

## Пример кода

```python
from imbalance_handling import ImbalanceHandler, TextAugmentation

# Создаем обработчик
handler = ImbalanceHandler(language='ru')
aug = handler.augmentation

# Пример текста
text = "Это пример текста для аугментации"

# Метод 1: EDA (рекомендуется)
augmented = aug.easy_data_augmentation(text, num_aug=4)
print("EDA:", augmented)

# Метод 2: Замена синонимами
synonym_text = aug.synonym_replacement(text)
print("Synonym:", synonym_text)

# Метод 3: Back translation (только если доступен)
# back_translated = aug.back_translation(text, load_model=True)
# print("Back translation:", back_translated)
```

## Вывод

**Для работы в России рекомендуется:**
- ✅ Использовать **EDA** как основной метод аугментации
- ✅ Использовать **SMOTE/ADASYN** для балансировки классов
- ❌ Избегать back translation, если внешние сервисы недоступны

EDA дает отличные результаты и не требует внешних зависимостей!

















