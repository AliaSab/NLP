import re
import html
import unicodedata
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class TextCleaner:
    """Модуль для очистки и нормализации текста"""
    
    def __init__(self, remove_stopwords: bool = True, language: str = 'russian'):
        self.should_remove_stopwords = remove_stopwords
        self.language = language
        self.stopwords = self._load_stopwords()
        
        # Регулярные выражения для очистки
        self.patterns = {
            'html_tags': re.compile(r'<[^>]+>'),
            'html_entities': re.compile(r'&[a-zA-Z0-9#]+;'),
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone_numbers': re.compile(r'(\+7|8)?[\s\-]?\(?[489][0-9]{2}\)?[\s\-]?[0-9]{3}[\s\-]?[0-9]{2}[\s\-]?[0-9]{2}'),
            'dates': re.compile(r'\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b'),
            'times': re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b'),
            'numbers': re.compile(r'\b\d+\b'),
            'special_chars': re.compile(r'[^\w\s\-.,!?;:()]'),
            'multiple_spaces': re.compile(r'\s+'),
            'multiple_punctuation': re.compile(r'([.!?]){2,}'),
            'quotes': re.compile(r'["""''«»]'),
            'dashes': re.compile(r'[–—]'),
        }
    
    def _load_stopwords(self) -> set:
        """Загрузка стоп-слов"""
        stopwords = set()
        
        if self.language == 'russian':
            # Базовый список русских стоп-слов
            russian_stopwords = {
                'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'
            }
            stopwords.update(russian_stopwords)
        
        return stopwords
    
    def clean_html(self, text: str) -> str:
        """Удаление HTML разметки и декодирование HTML сущностей"""
        if not text:
            return ""
        
        # Декодирование HTML сущностей
        text = html.unescape(text)
        
        # Удаление HTML тегов
        text = self.patterns['html_tags'].sub('', text)
        
        # Удаление оставшихся HTML сущностей
        text = self.patterns['html_entities'].sub('', text)
        
        return text
    
    def clean_special_characters(self, text: str) -> str:
        """Очистка от специальных символов и нормализация"""
        if not text:
            return ""
        
        # Нормализация Unicode
        text = unicodedata.normalize('NFKC', text)
        
        # Замена кавычек на стандартные
        text = self.patterns['quotes'].sub('"', text)
        
        # Замена тире на стандартные
        text = self.patterns['dashes'].sub('-', text)
        
        # Удаление специальных символов (оставляем только буквы, цифры, пробелы и пунктуацию)
        text = self.patterns['special_chars'].sub('', text)
        
        return text
    
    def normalize_whitespace(self, text: str) -> str:
        """Нормализация пробельных символов"""
        if not text:
            return ""
        
        # Замена множественных пробелов на одинарные
        text = self.patterns['multiple_spaces'].sub(' ', text)
        
        # Удаление пробелов в начале и конце
        text = text.strip()
        
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """Нормализация пунктуации"""
        if not text:
            return ""
        
        # Нормализация множественной пунктуации
        text = self.patterns['multiple_punctuation'].sub(r'\1', text)
        
        return text
    
    def remove_urls_and_emails(self, text: str) -> str:
        """Удаление URL-адресов и email"""
        if not text:
            return ""
        
        # Удаление URL
        text = self.patterns['urls'].sub('', text)
        
        # Удаление email
        text = self.patterns['emails'].sub('', text)
        
        return text
    
    def remove_phone_numbers(self, text: str) -> str:
        """Удаление телефонных номеров"""
        if not text:
            return ""
        
        text = self.patterns['phone_numbers'].sub('', text)
        return text
    
    def remove_dates_and_times(self, text: str) -> str:
        """Удаление дат и времени"""
        if not text:
            return ""
        
        # Удаление дат
        text = self.patterns['dates'].sub('', text)
        
        # Удаление времени
        text = self.patterns['times'].sub('', text)
        
        return text
    
    def remove_number_tokens(self, text: str) -> str:
        """Удаление чисел"""
        if not text:
            return ""
        
        text = self.patterns['numbers'].sub('', text)
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """Удаление стоп-слов"""
        if not text or not self.should_remove_stopwords:
            return text
        
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stopwords]
        
        return ' '.join(filtered_words)
    
    def to_lowercase(self, text: str) -> str:
        """Приведение к нижнему регистру"""
        if not text:
            return ""
        
        return text.lower()
    
    def clean_text(self, text: str, 
                   remove_html: bool = True,
                   remove_urls: bool = True,
                   remove_phones: bool = True,
                   remove_dates: bool = True,
                   remove_numbers: bool = False,
                   normalize_whitespace: bool = True,
                   normalize_punctuation: bool = True,
                   to_lowercase: bool = False,
                   remove_stopwords: bool = None) -> str:
        """
        Основная функция очистки текста
        
        Args:
            text: Исходный текст
            remove_html: Удалять HTML разметку
            remove_urls: Удалять URL и email
            remove_phones: Удалять телефонные номера
            remove_dates: Удалять даты и время
            remove_numbers: Удалять числа
            normalize_whitespace: Нормализовать пробелы
            normalize_punctuation: Нормализовать пунктуацию
            to_lowercase: Приводить к нижнему регистру
            remove_stopwords: Удалять стоп-слова (None = использовать настройку по умолчанию)
        
        Returns:
            Очищенный текст
        """
        if not text:
            return ""
        
        # HTML очистка
        if remove_html:
            text = self.clean_html(text)
        
        # Удаление URL и email
        if remove_urls:
            text = self.remove_urls_and_emails(text)
        
        # Удаление телефонных номеров
        if remove_phones:
            text = self.remove_phone_numbers(text)
        
        # Удаление дат и времени
        if remove_dates:
            text = self.remove_dates_and_times(text)
        
        # Удаление чисел
        if remove_numbers:
            text = self.remove_number_tokens(text)
        
        # Очистка специальных символов
        text = self.clean_special_characters(text)
        
        # Нормализация пробелов
        if normalize_whitespace:
            text = self.normalize_whitespace(text)
        
        # Нормализация пунктуации
        if normalize_punctuation:
            text = self.normalize_punctuation(text)
        
        # Приведение к нижнему регистру
        if to_lowercase:
            text = self.to_lowercase(text)
        
        # Удаление стоп-слов
        if remove_stopwords is not None:
            if remove_stopwords:
                text = self.remove_stopwords(text)
        elif self.should_remove_stopwords:
            text = self.remove_stopwords(text)
        
        return text
    
    def clean_article(self, article: Dict[str, Any], 
                     clean_title: bool = True,
                     clean_text: bool = True,
                     **kwargs) -> Dict[str, Any]:
        """
        Очистка статьи (заголовок + текст)
        
        Args:
            article: Словарь с данными статьи
            clean_title: Очищать заголовок
            clean_text: Очищать текст
            **kwargs: Параметры для clean_text
        
        Returns:
            Очищенная статья
        """
        cleaned_article = article.copy()
        
        if clean_title and 'title' in article:
            cleaned_article['title'] = self.clean_text(article['title'], **kwargs)
        
        if clean_text and 'text' in article:
            cleaned_article['text'] = self.clean_text(article['text'], **kwargs)
        
        return cleaned_article
    
    def batch_clean(self, articles: List[Dict[str, Any]], 
                   clean_title: bool = True,
                   clean_text: bool = True,
                   **kwargs) -> List[Dict[str, Any]]:
        """
        Пакетная очистка статей
        
        Args:
            articles: Список статей
            clean_title: Очищать заголовки
            clean_text: Очищать тексты
            **kwargs: Параметры для clean_text
        
        Returns:
            Список очищенных статей
        """
        cleaned_articles = []
        
        for i, article in enumerate(articles):
            try:
                cleaned_article = self.clean_article(article, clean_title, clean_text, **kwargs)
                cleaned_articles.append(cleaned_article)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Очищено {i + 1}/{len(articles)} статей")
                    
            except Exception as e:
                logger.error(f"Ошибка при очистке статьи {i}: {e}")
                continue
        
        logger.info(f"Очистка завершена. Обработано {len(cleaned_articles)} из {len(articles)} статей")
        return cleaned_articles

def main():
    """Пример использования TextCleaner"""
    # Создание экземпляра очистителя
    cleaner = TextCleaner(remove_stopwords=True, language='russian')
    
    # Пример текста с HTML и мусором
    sample_text = """
    <div class="article">
        <h1>Заголовок статьи</h1>
        <p>Это пример текста с <strong>HTML</strong> разметкой.</p>
        <p>Ссылка: https://example.com и email: test@example.com</p>
        <p>Телефон: +7 (495) 123-45-67</p>
        <p>Дата: 15.12.2023, время: 14:30</p>
        <p>Числа: 123, 456, 789</p>
        <p>Множественные    пробелы    и    знаки    препинания!!!</p>
    </div>
    """
    
    print("Исходный текст:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    # Очистка с разными параметрами
    print("1. Базовая очистка:")
    cleaned1 = cleaner.clean_text(sample_text)
    print(cleaned1)
    print("\n" + "="*50 + "\n")
    
    print("2. Очистка с удалением чисел:")
    cleaned2 = cleaner.clean_text(sample_text, remove_numbers=True)
    print(cleaned2)
    print("\n" + "="*50 + "\n")
    
    print("3. Очистка с приведением к нижнему регистру:")
    cleaned3 = cleaner.clean_text(sample_text, to_lowercase=True)
    print(cleaned3)
    print("\n" + "="*50 + "\n")
    
    print("4. Полная очистка с удалением стоп-слов:")
    cleaned4 = cleaner.clean_text(sample_text, to_lowercase=True, remove_stopwords=True)
    print(cleaned4)

if __name__ == "__main__":
    main()

