"""
Скрипт для подготовки экспериментального корпуса текстов для классификации.
Этап 1: Разметка данных для бинарной, многоклассовой и многометочной классификации.
"""

import json
import os
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import random
import numpy as np


# Ключевые слова для определения тональности
POSITIVE_KEYWORDS = [
    'успех', 'победа', 'достижение', 'рост', 'увеличение', 'прибыль', 'развитие',
    'инвестиции', 'сотрудничество', 'партнерство', 'награда', 'премия', 'открытие',
    'новый', 'прогресс', 'улучшение', 'результат', 'эффективность', 'лидер',
    'чемпион', 'рекорд', 'триумф', 'достижение', 'прорыв', 'инновация'
]

NEGATIVE_KEYWORDS = [
    'банкрот', 'кризис', 'падение', 'снижение', 'убыток', 'проблема', 'конфликт',
    'война', 'атака', 'террор', 'авария', 'катастрофа', 'смерть', 'болезнь',
    'рецессия', 'дефолт', 'долг', 'задержка', 'отмена', 'закрытие', 'увольнение',
    'суд', 'арест', 'обвинение', 'штраф', 'санкции', 'конфликт', 'протест'
]

# Ключевые слова для определения категорий
POLITICS_KEYWORDS = [
    'президент', 'правительство', 'министр', 'депутат', 'парламент', 'выборы',
    'голосование', 'политика', 'партия', 'оппозиция', 'закон', 'законопроект',
    'санкции', 'дипломатия', 'посол', 'встреча', 'саммит', 'переговоры',
    'государство', 'власть', 'администрация', 'кремль', 'дума', 'совет'
]

ECONOMY_KEYWORDS = [
    'экономика', 'рынок', 'акции', 'биржа', 'инвестиции', 'банк', 'финансы',
    'валюта', 'рубль', 'доллар', 'евро', 'инфляция', 'ввп', 'бюджет',
    'компания', 'бизнес', 'прибыль', 'убыток', 'доход', 'расход', 'налог',
    'торговля', 'экспорт', 'импорт', 'нефть', 'газ', 'цена', 'стоимость',
    'банкрот', 'кредит', 'заем', 'долг', 'облигации', 'фонд'
]

SPORT_KEYWORDS = [
    'спорт', 'футбол', 'хоккей', 'баскетбол', 'теннис', 'олимпиада', 'чемпионат',
    'матч', 'игра', 'команда', 'игрок', 'тренер', 'стадион', 'соревнование',
    'победа', 'поражение', 'гол', 'очко', 'медаль', 'чемпион', 'рекорд',
    'лига', 'турнир', 'финал', 'полуфинал', 'квалификация'
]

CULTURE_KEYWORDS = [
    'культура', 'искусство', 'театр', 'кино', 'фильм', 'музей', 'выставка',
    'концерт', 'музыка', 'песня', 'альбом', 'книга', 'писатель', 'поэт',
    'художник', 'актер', 'режиссер', 'фестиваль', 'премия', 'награда',
    'литература', 'живопись', 'скульптура', 'архитектура', 'галерея'
]


def determine_sentiment(text: str) -> str:
    """
    Определяет тональность текста на основе ключевых слов.
    
    Args:
        text: Текст для анализа
        
    Returns:
        'positive' или 'negative'
    """
    text_lower = text.lower()
    
    positive_count = sum(1 for keyword in POSITIVE_KEYWORDS if keyword in text_lower)
    negative_count = sum(1 for keyword in NEGATIVE_KEYWORDS if keyword in text_lower)
    
    # Если есть слова про банкротство, суды и т.д., скорее всего негативная
    if any(word in text_lower for word in ['банкрот', 'суд', 'арест', 'обвинение', 'реализация имущества']):
        return 'negative'
    
    # Если есть слова про развитие, сотрудничество, рост - позитивная
    if any(word in text_lower for word in ['сотрудничество', 'развитие', 'рост', 'успех', 'победа', 'достижение']):
        if positive_count >= 2:
            return 'positive'
    
    if positive_count > negative_count and positive_count > 0:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        # По умолчанию нейтральная/негативная для новостей
        return 'negative'


def determine_category(text: str) -> str:
    """
    Определяет основную категорию текста.
    
    Args:
        text: Текст для анализа
        
    Returns:
        'politics', 'economy', 'sport', 'culture'
    """
    text_lower = text.lower()
    
    category_scores = {
        'politics': sum(1 for keyword in POLITICS_KEYWORDS if keyword in text_lower),
        'economy': sum(1 for keyword in ECONOMY_KEYWORDS if keyword in text_lower),
        'sport': sum(1 for keyword in SPORT_KEYWORDS if keyword in text_lower),
        'culture': sum(1 for keyword in CULTURE_KEYWORDS if keyword in text_lower)
    }
    
    # Если есть явные указания на категорию в тексте
    if 'посол' in text_lower or 'дипломат' in text_lower or 'министр' in text_lower:
        category_scores['politics'] += 2
    if 'авиасообщение' in text_lower or 'туризм' in text_lower:
        category_scores['economy'] += 1
        category_scores['culture'] += 1
    
    # Если нет совпадений, определяем по контексту
    if sum(category_scores.values()) == 0:
        # Если есть слова про суды, банкротства - экономика
        if any(word in text_lower for word in ['суд', 'банкрот', 'арбитражный']):
            return 'economy'
        # Если есть слова про правительство, министерство - политика
        if any(word in text_lower for word in ['правительство', 'министерство', 'государство']):
            return 'politics'
        # По умолчанию экономика для новостей
        return 'economy'
    
    # Возвращаем категорию с максимальным счетом
    return max(category_scores, key=category_scores.get)


def determine_multilabel_categories(text: str) -> List[str]:
    """
    Определяет несколько категорий для многометочной классификации.
    
    Args:
        text: Текст для анализа
        
    Returns:
        Список категорий
    """
    text_lower = text.lower()
    
    categories = []
    
    # Проверяем каждую категорию (более мягкие условия для многометочной)
    politics_score = sum(1 for keyword in POLITICS_KEYWORDS if keyword in text_lower)
    economy_score = sum(1 for keyword in ECONOMY_KEYWORDS if keyword in text_lower)
    sport_score = sum(1 for keyword in SPORT_KEYWORDS if keyword in text_lower)
    culture_score = sum(1 for keyword in CULTURE_KEYWORDS if keyword in text_lower)
    
    # Добавляем категории, если есть хотя бы одно совпадение
    if politics_score > 0 or any(word in text_lower for word in ['правительство', 'министр', 'посол', 'дипломат']):
        categories.append('politics')
    
    if economy_score > 0 or any(word in text_lower for word in ['банкрот', 'суд', 'арбитражный', 'финанс', 'экономик']):
        categories.append('economy')
    
    if sport_score > 0:
        categories.append('sport')
    
    if culture_score > 0 or any(word in text_lower for word in ['туризм', 'авиасообщение', 'культур']):
        categories.append('culture')
    
    # Если нет категорий, добавляем экономику по умолчанию
    if not categories:
        categories.append('economy')
    
    # Если есть суды/банкротства, добавляем политику, если её ещё нет
    if 'economy' in categories and any(word in text_lower for word in ['суд', 'арбитражный']) and 'politics' not in categories:
        # С вероятностью 30% добавляем политику для разнообразия
        if random.random() < 0.3:
            categories.append('politics')
    
    return categories


def load_corpus(input_file: str) -> List[Dict]:
    """
    Загружает корпус из JSONL файла.
    
    Args:
        input_file: Путь к входному файлу
        
    Returns:
        Список словарей с данными
    """
    articles = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles


def annotate_articles(articles: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Размечает статьи для трех типов задач классификации.
    
    Args:
        articles: Список статей
        
    Returns:
        Кортеж из трех списков: бинарная, многоклассовая, многометочная
    """
    binary_data = []
    multiclass_data = []
    multilabel_data = []
    
    for article in articles:
        title = article.get('title', '')
        text = article.get('text', '')
        full_text = f"{title} {text}".strip()
        
        # Пропускаем слишком короткие тексты
        if len(full_text.split()) < 10:
            continue
        
        # Бинарная классификация (тональность)
        sentiment = determine_sentiment(full_text)
        binary_data.append({
            'title': title,
            'text': text,
            'sentiment': sentiment
        })
        
        # Многоклассовая классификация (одна категория)
        category = determine_category(full_text)
        multiclass_data.append({
            'title': title,
            'text': text,
            'category': category
        })
        
        # Многометочная классификация (несколько категорий)
        categories = determine_multilabel_categories(full_text)
        multilabel_data.append({
            'title': title,
            'text': text,
            'categories': categories
        })
    
    return binary_data, multiclass_data, multilabel_data


def balance_classes(data: List[Dict], label_key: str, min_samples: int = 10000) -> List[Dict]:
    """
    Балансирует классы, ограничивая дисбаланс до 3:1.
    
    Args:
        data: Список размеченных данных
        label_key: Ключ для метки ('sentiment', 'category' или 'categories')
        min_samples: Минимальное количество образцов (используется только для информации)
        
    Returns:
        Сбалансированный список данных
    """
    # Подсчитываем распределение классов
    if label_key == 'categories':
        # Для многометочной классификации считаем по каждой категории
        label_counts = defaultdict(int)
        for item in data:
            for cat in item[label_key]:
                label_counts[cat] += 1
    else:
        label_counts = Counter(item[label_key] for item in data)
    
    print(f"\nРаспределение классов до балансировки:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Определяем минимальное и максимальное количество
    if label_key == 'categories':
        min_count = min(label_counts.values()) if label_counts else 0
        max_count = max(label_counts.values()) if label_counts else 0
    else:
        min_count = min(label_counts.values()) if label_counts else 0
        max_count = max(label_counts.values()) if label_counts else 0
    
    if min_count == 0:
        print("Предупреждение: некоторые классы пусты!")
        return data
    
    # Если данных очень мало, не балансируем агрессивно
    total_samples = len(data)
    if total_samples < 100:
        print(f"⚠️ Внимание: данных очень мало ({total_samples} образцов). Балансировка будет менее агрессивной.")
        # Для малых выборок используем более мягкое ограничение (5:1 вместо 3:1)
        max_allowed = min_count * 5
    else:
        # Ограничиваем дисбаланс до 3:1
        max_allowed = min_count * 3
    
    if max_allowed == 0:
        max_allowed = max_count
    
    if label_key == 'categories':
        # Для многометочной классификации балансируем по каждой категории
        balanced_data = []
        category_data = defaultdict(list)
        
        for item in data:
            for cat in item[label_key]:
                category_data[cat].append(item)
        
        # Ограничиваем каждую категорию
        limited_category_data = {}
        for cat, items in category_data.items():
            if len(items) > max_allowed:
                limited_category_data[cat] = random.sample(items, max_allowed)
            else:
                limited_category_data[cat] = items
        
        # Объединяем данные, избегая дубликатов
        seen = set()
        for items in limited_category_data.values():
            for item in items:
                # Используем комбинацию title+text для идентификации
                item_id = (item.get('title', ''), item.get('text', ''))
                if item_id not in seen:
                    balanced_data.append(item)
                    seen.add(item_id)
        
    else:
        # Для бинарной и многоклассовой классификации
        balanced_data = []
        class_data = defaultdict(list)
        
        for item in data:
            class_data[item[label_key]].append(item)
        
        # Ограничиваем каждый класс
        for label, items in class_data.items():
            if len(items) > max_allowed:
                balanced_data.extend(random.sample(items, max_allowed))
            else:
                balanced_data.extend(items)
    
    # Перемешиваем
    random.shuffle(balanced_data)
    
    print(f"\nРаспределение классов после балансировки:")
    if label_key == 'categories':
        new_label_counts = defaultdict(int)
        for item in balanced_data:
            for cat in item[label_key]:
                new_label_counts[cat] += 1
        for label, count in sorted(new_label_counts.items()):
            print(f"  {label}: {count}")
    else:
        new_label_counts = Counter(item[label_key] for item in balanced_data)
        for label, count in sorted(new_label_counts.items()):
            print(f"  {label}: {count}")
    
    print(f"Всего образцов после балансировки: {len(balanced_data)}")
    if len(balanced_data) < min_samples:
        print(f"⚠️ Предупреждение: получено {len(balanced_data)} образцов, требуется минимум {min_samples}")
        if len(balanced_data) < 100:
            print(f"⚠️ КРИТИЧЕСКОЕ ПРЕДУПРЕЖДЕНИЕ: Данных недостаточно для качественного обучения!")
            print(f"   Рекомендуется увеличить исходный корпус до минимум 5000-10000 статей.")
    
    return balanced_data


def split_data(data: List[Dict], label_key: str, train_ratio: float = 0.7, 
               val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Разделяет данные на train/validation/test с сохранением стратификации.
    
    Args:
        data: Список размеченных данных
        label_key: Ключ для метки
        train_ratio: Доля обучающей выборки
        val_ratio: Доля валидационной выборки
        test_ratio: Доля тестовой выборки
        
    Returns:
        Кортеж из трех списков: train, validation, test
    """
    # Извлекаем метки для стратификации
    if label_key == 'categories':
        # Для многометочной классификации используем первую категорию
        labels = [item[label_key][0] if item[label_key] else 'unknown' for item in data]
    else:
        labels = [item[label_key] for item in data]
    
    # Группируем данные по меткам для стратификации
    label_groups = defaultdict(list)
    for item, label in zip(data, labels):
        label_groups[label].append(item)
    
    # Разделяем каждую группу пропорционально
    train_data = []
    val_data = []
    test_data = []
    
    for label, items in label_groups.items():
        random.shuffle(items)
        n = len(items)
        
        # Гарантируем минимум 1 образец в каждой выборке, если данных достаточно
        if n >= 3:
            train_end = max(1, int(n * train_ratio))
            val_end = train_end + max(1, int(n * val_ratio))
            # Убеждаемся, что test тоже получит хотя бы 1 образец
            if val_end >= n:
                val_end = n - 1
                train_end = max(1, val_end - 1)
        else:
            # Если данных очень мало, распределяем вручную
            if n == 1:
                train_end = 1
                val_end = 1
            elif n == 2:
                train_end = 1
                val_end = 2
            else:  # n == 0 (не должно быть, но на всякий случай)
                continue
        
        train_data.extend(items[:train_end])
        val_data.extend(items[train_end:val_end])
        test_data.extend(items[val_end:])
    
    # Перемешиваем финальные выборки
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data


def save_jsonl(data: List[Dict], output_file: str):
    """
    Сохраняет данные в JSONL формат.
    
    Args:
        data: Список словарей для сохранения
        output_file: Путь к выходному файлу
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    """Основная функция для подготовки корпуса."""
    # Пути к файлам
    input_file = 'kommersant_articles_processed.jsonl'
    output_dir = 'corpus'
    
    print("Загрузка корпуса...")
    articles = load_corpus(input_file)
    print(f"Загружено статей: {len(articles)}")
    
    if len(articles) < 1000:
        print(f"\n⚠️ ВНИМАНИЕ: Исходных данных очень мало ({len(articles)} статей)!")
        print("Для качественного обучения рекомендуется не менее 5000-10000 статей.")
        print("Результаты могут быть нерепрезентативными.\n")
    
    print("\nРазметка статей...")
    binary_data, multiclass_data, multilabel_data = annotate_articles(articles)
    
    print(f"\nБинарная классификация: {len(binary_data)} статей")
    print(f"Многоклассовая классификация: {len(multiclass_data)} статей")
    print(f"Многометочная классификация: {len(multilabel_data)} статей")
    
    if len(binary_data) < 100:
        print(f"\n⚠️ ВНИМАНИЕ: После разметки осталось очень мало данных для бинарной классификации!")
    if len(multiclass_data) < 100:
        print(f"⚠️ ВНИМАНИЕ: После разметки осталось очень мало данных для многоклассовой классификации!")
    
    # Балансировка классов
    print("\n" + "="*50)
    print("Балансировка классов для бинарной классификации...")
    binary_balanced = balance_classes(binary_data, 'sentiment', min_samples=10000)
    
    print("\n" + "="*50)
    print("Балансировка классов для многоклассовой классификации...")
    multiclass_balanced = balance_classes(multiclass_data, 'category', min_samples=10000)
    
    print("\n" + "="*50)
    print("Балансировка классов для многометочной классификации...")
    multilabel_balanced = balance_classes(multilabel_data, 'categories', min_samples=10000)
    
    # Разделение на train/val/test
    print("\n" + "="*50)
    print("Разделение данных на train/validation/test...")
    
    # Бинарная классификация
    binary_train, binary_val, binary_test = split_data(binary_balanced, 'sentiment')
    print(f"\nБинарная классификация:")
    print(f"  Train: {len(binary_train)}")
    print(f"  Validation: {len(binary_val)}")
    print(f"  Test: {len(binary_test)}")
    
    # Многоклассовая классификация
    multiclass_train, multiclass_val, multiclass_test = split_data(multiclass_balanced, 'category')
    print(f"\nМногоклассовая классификация:")
    print(f"  Train: {len(multiclass_train)}")
    print(f"  Validation: {len(multiclass_val)}")
    print(f"  Test: {len(multiclass_test)}")
    
    # Многометочная классификация
    multilabel_train, multilabel_val, multilabel_test = split_data(multilabel_balanced, 'categories')
    print(f"\nМногометочная классификация:")
    print(f"  Train: {len(multilabel_train)}")
    print(f"  Validation: {len(multilabel_val)}")
    print(f"  Test: {len(multilabel_test)}")
    
    # Сохранение данных
    print("\n" + "="*50)
    print("Сохранение данных...")
    
    # Бинарная классификация
    save_jsonl(binary_train, f'{output_dir}/binary_classification/train.jsonl')
    save_jsonl(binary_val, f'{output_dir}/binary_classification/validation.jsonl')
    save_jsonl(binary_test, f'{output_dir}/binary_classification/test.jsonl')
    print("Бинарная классификация сохранена")
    
    # Многоклассовая классификация
    save_jsonl(multiclass_train, f'{output_dir}/multiclass_classification/train.jsonl')
    save_jsonl(multiclass_val, f'{output_dir}/multiclass_classification/validation.jsonl')
    save_jsonl(multiclass_test, f'{output_dir}/multiclass_classification/test.jsonl')
    print("Многоклассовая классификация сохранена")
    
    # Многометочная классификация
    save_jsonl(multilabel_train, f'{output_dir}/multilabel_classification/train.jsonl')
    save_jsonl(multilabel_val, f'{output_dir}/multilabel_classification/validation.jsonl')
    save_jsonl(multilabel_test, f'{output_dir}/multilabel_classification/test.jsonl')
    print("Многометочная классификация сохранена")
    
    print("\n" + "="*50)
    print("Готово! Корпус успешно подготовлен.")


if __name__ == '__main__':
    random.seed(42)
    np.random.seed(42)
    main()

