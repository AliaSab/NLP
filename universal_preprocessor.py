import os
os.environ['TCL_LIBRARY'] = "C:/Program Files/Python313/tcl/tcl8.6"
os.environ['TK_LIBRARY'] = "C:/Program Files/Python313/tcl/tk8.6"
import re
import json
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Конфигурация для предобработки текста"""
    # Замена на токены
    replace_numbers: bool = True
    replace_urls: bool = True
    replace_emails: bool = True
    replace_phones: bool = True
    replace_dates: bool = True
    replace_times: bool = True
    replace_currencies: bool = True
    
    # Нормализация
    normalize_punctuation: bool = True
    normalize_quotes: bool = True
    normalize_dashes: bool = True
    normalize_spaces: bool = True
    
    # Обработка сокращений
    expand_abbreviations: bool = True
    expand_contractions: bool = True
    
    # Дополнительные настройки
    to_lowercase: bool = False
    remove_extra_punctuation: bool = True
    preserve_sentence_structure: bool = True

class UniversalPreprocessor:
    """Универсальный модуль предобработки текста"""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None, language: str = 'russian'):
        self.config = config or PreprocessingConfig()
        self.language = language
        
        # Словари сокращений
        self.abbreviations = self._load_abbreviations()
        self.contractions = self._load_contractions()
        
        # Регулярные выражения
        self.patterns = self._init_patterns()
        
        # Специальные токены
        self.tokens = {
            'NUM': '<NUM>',
            'URL': '<URL>',
            'EMAIL': '<EMAIL>',
            'PHONE': '<PHONE>',
            'DATE': '<DATE>',
            'TIME': '<TIME>',
            'CURRENCY': '<CURRENCY>',
            'UNK': '<UNK>'
        }
    
    def _load_abbreviations(self) -> Dict[str, str]:
        """Загрузка словаря сокращений"""
        abbreviations = {}
        
        if self.language == 'russian':
            abbreviations = {
                # Общие сокращения
                'т.е.': 'то есть',
                'т.д.': 'так далее',
                'т.п.': 'тому подобное',
                'и т.д.': 'и так далее',
                'и т.п.': 'и тому подобное',
                'и др.': 'и другие',
                'и пр.': 'и прочие',
                'и т.о.': 'и тому подобное',
                'и т.с.': 'и так сказать',
                
                # Временные сокращения
                'г.': 'год',
                'гг.': 'годы',
                'в.': 'век',
                'вв.': 'века',
                'н.э.': 'нашей эры',
                'до н.э.': 'до нашей эры',
                'мин.': 'минут',
                'сек.': 'секунд',
                'час.': 'часов',
                'дн.': 'дней',
                'нед.': 'недель',
                'мес.': 'месяцев',
                
                # Географические сокращения
                'г.': 'город',
                'с.': 'село',
                'д.': 'деревня',
                'п.': 'поселок',
                'р.': 'река',
                'оз.': 'озеро',
                'м.': 'место',
                'обл.': 'область',
                'край': 'край',
                'респ.': 'республика',
                'р-н': 'район',
                'ул.': 'улица',
                'пр.': 'проспект',
                'пер.': 'переулок',
                'пл.': 'площадь',
                'наб.': 'набережная',
                
                # Организационные сокращения
                'ООО': 'общество с ограниченной ответственностью',
                'ЗАО': 'закрытое акционерное общество',
                'ОАО': 'открытое акционерное общество',
                'ИП': 'индивидуальный предприниматель',
                'ОООО': 'общество с ограниченной ответственностью',
                'АО': 'акционерное общество',
                'ПАО': 'публичное акционерное общество',
                'ТОО': 'товарищество с ограниченной ответственностью',
                
                # Научные сокращения
                'др.': 'доктор',
                'проф.': 'профессор',
                'доц.': 'доцент',
                'акад.': 'академик',
                'чл.-корр.': 'член-корреспондент',
                'к.т.н.': 'кандидат технических наук',
                'д.т.н.': 'доктор технических наук',
                'к.ф.-м.н.': 'кандидат физико-математических наук',
                'д.ф.-м.н.': 'доктор физико-математических наук',
                
                # Военные сокращения
                'ген.': 'генерал',
                'полк.': 'полковник',
                'подполк.': 'подполковник',
                'майор': 'майор',
                'кап.': 'капитан',
                'ст. лейт.': 'старший лейтенант',
                'лейт.': 'лейтенант',
                'мл. лейт.': 'младший лейтенант',
                
                # Медицинские сокращения
                'д-р': 'доктор',
                'мед.': 'медицинский',
                'ст.': 'старший',
                'мл.': 'младший',
                'гл.': 'главный',
                'зам.': 'заместитель',
                'зав.': 'заведующий',
                'нач.': 'начальник',
                
                # Технические сокращения
                'т.': 'том',
                'ч.': 'часть',
                'гл.': 'глава',
                'п.': 'пункт',
                'ст.': 'статья',
                'разд.': 'раздел',
                'подразд.': 'подраздел',
                'пп.': 'подпункты',
                'рис.': 'рисунок',
                'табл.': 'таблица',
                'стр.': 'страница',
                'с.': 'страница',
                'стр.': 'страницы',
                'с.': 'страницы',
            }
        
        return abbreviations
    
    def _load_contractions(self) -> Dict[str, str]:
        """Загрузка словаря сокращений (стяжений)"""
        contractions = {}
        
        if self.language == 'russian':
            contractions = {
                # Стяжения в русском языке
                'нельзя': 'нельзя',
                'нельзя': 'нельзя',
                'нельзя': 'нельзя',
            }
        
        return contractions
    
    def _init_patterns(self) -> Dict[str, re.Pattern]:
        """Инициализация регулярных выражений"""
        patterns = {
            # Числа
            'numbers': re.compile(r'\b\d+(?:[.,]\d+)?\b'),
            'ordinal_numbers': re.compile(r'\b\d+(?:-й|-я|-е|-го|-му|-м|-х|-ми|-ми)\b'),
            'mixed_words': re.compile(r'\b\w*\d+\w*\b'),  # Слова, содержащие цифры
            
            # URL и email
            'urls': re.compile(r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            
            # Телефоны
            'phones': re.compile(r'(?:\+7|8)?[\s\-]?\(?[489][0-9]{2}\)?[\s\-]?[0-9]{3}[\s\-]?[0-9]{2}[\s\-]?[0-9]{2}'),
            
            # Даты
            'dates_dd_mm_yyyy': re.compile(r'\b\d{1,2}[./]\d{1,2}[./]\d{4}\b'),
            'dates_dd_mm_yy': re.compile(r'\b\d{1,2}[./]\d{1,2}[./]\d{2}\b'),
            'dates_yyyy_mm_dd': re.compile(r'\b\d{4}[./-]\d{1,2}[./-]\d{1,2}\b'),
            'dates_text': re.compile(r'\b\d{1,2}\s+(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s+\d{4}\b'),
            
            # Время
            'times': re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm|утра|дня|вечера|ночи)?\b'),
            
            # Валюты
            'currencies': re.compile(r'\b\d+(?:[.,]\d+)?\s*(?:руб|рублей|рубля|₽|дол|долларов|доллара|\$|евро|€|тенге|сом|гривен|грн)\b'),
            
            # Пунктуация
            'multiple_punctuation': re.compile(r'([.!?]){2,}'),
            'quotes': re.compile(r'["""''«»]'),
            'dashes': re.compile(r'[–—]'),
            'ellipsis': re.compile(r'\.{3,}'),
            
            # Пробелы
            'multiple_spaces': re.compile(r'\s+'),
            'leading_trailing_spaces': re.compile(r'^\s+|\s+$'),
        }
        
        return patterns
    
    def replace_numbers(self, text: str) -> str:
        """Замена чисел на токен <NUM>"""
        if not self.config.replace_numbers:
            return text
        
        # Замена обычных чисел
        text = self.patterns['numbers'].sub(self.tokens['NUM'], text)
        
        # Замена порядковых числительных
        text = self.patterns['ordinal_numbers'].sub(self.tokens['NUM'], text)
        
        # Замена слов, содержащих цифры
        text = self.patterns['mixed_words'].sub(self.tokens['NUM'], text)
        
        return text
    
    def replace_urls_and_emails(self, text: str) -> str:
        """Замена URL и email на токены"""
        if self.config.replace_urls:
            text = self.patterns['urls'].sub(self.tokens['URL'], text)
        
        if self.config.replace_emails:
            text = self.patterns['emails'].sub(self.tokens['EMAIL'], text)
        
        return text
    
    def replace_phones(self, text: str) -> str:
        """Замена телефонных номеров на токен <PHONE>"""
        if not self.config.replace_phones:
            return text
        
        text = self.patterns['phones'].sub(self.tokens['PHONE'], text)
        return text
    
    def replace_dates(self, text: str) -> str:
        """Замена дат на токен <DATE>"""
        if not self.config.replace_dates:
            return text
        
        # Различные форматы дат
        text = self.patterns['dates_dd_mm_yyyy'].sub(self.tokens['DATE'], text)
        text = self.patterns['dates_dd_mm_yy'].sub(self.tokens['DATE'], text)
        text = self.patterns['dates_yyyy_mm_dd'].sub(self.tokens['DATE'], text)
        text = self.patterns['dates_text'].sub(self.tokens['DATE'], text)
        
        return text
    
    def replace_times(self, text: str) -> str:
        """Замена времени на токен <TIME>"""
        if not self.config.replace_times:
            return text
        
        text = self.patterns['times'].sub(self.tokens['TIME'], text)
        return text
    
    def replace_currencies(self, text: str) -> str:
        """Замена валют на токен <CURRENCY>"""
        if not self.config.replace_currencies:
            return text
        
        text = self.patterns['currencies'].sub(self.tokens['CURRENCY'], text)
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """Нормализация пунктуации"""
        if not self.config.normalize_punctuation:
            return text
        
        # Нормализация множественной пунктуации
        text = self.patterns['multiple_punctuation'].sub(r'\1', text)
        
        # Нормализация многоточия
        text = self.patterns['ellipsis'].sub('...', text)
        
        return text
    
    def normalize_quotes_and_dashes(self, text: str) -> str:
        """Нормализация кавычек и тире"""
        if self.config.normalize_quotes:
            text = self.patterns['quotes'].sub('"', text)
        
        if self.config.normalize_dashes:
            text = self.patterns['dashes'].sub('-', text)
        
        return text
    
    def normalize_spaces(self, text: str) -> str:
        """Нормализация пробелов"""
        if not self.config.normalize_spaces:
            return text
        
        # Замена множественных пробелов на одинарные
        text = self.patterns['multiple_spaces'].sub(' ', text)
        
        # Удаление пробелов в начале и конце
        text = self.patterns['leading_trailing_spaces'].sub('', text)
        
        return text
    
    def expand_abbreviations(self, text: str) -> str:
        """Расшифровка сокращений"""
        if not self.config.expand_abbreviations:
            return text
        
        for abbrev, expansion in self.abbreviations.items():
            # Используем word boundary для точного совпадения
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def expand_contractions(self, text: str) -> str:
        """Расшифровка стяжений"""
        if not self.config.expand_contractions:
            return text
        
        for contraction, expansion in self.contractions.items():
            pattern = r'\b' + re.escape(contraction) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Основная функция предобработки текста"""
        if not text:
            return ""
        
        # Замена на токены
        text = self.replace_numbers(text)
        text = self.replace_urls_and_emails(text)
        text = self.replace_phones(text)
        text = self.replace_dates(text)
        text = self.replace_times(text)
        text = self.replace_currencies(text)
        
        # Расшифровка сокращений
        text = self.expand_abbreviations(text)
        text = self.expand_contractions(text)
        
        # Нормализация
        text = self.normalize_punctuation(text)
        text = self.normalize_quotes_and_dashes(text)
        text = self.normalize_spaces(text)
        
        # Приведение к нижнему регистру
        if self.config.to_lowercase:
            text = text.lower()
        
        return text
    
    def preprocess_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Предобработка статьи"""
        processed_article = article.copy()
        
        if 'title' in article:
            processed_article['title'] = self.preprocess_text(article['title'])
        
        if 'text' in article:
            processed_article['text'] = self.preprocess_text(article['text'])
        
        return processed_article
    
    def batch_preprocess(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Пакетная предобработка статей"""
        processed_articles = []
        
        for i, article in enumerate(articles):
            try:
                processed_article = self.preprocess_article(article)
                processed_articles.append(processed_article)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"Предобработано {i + 1}/{len(articles)} статей")
                    
            except Exception as e:
                logger.error(f"Ошибка при предобработке статьи {i}: {e}")
                continue
        
        logger.info(f"Предобработка завершена. Обработано {len(processed_articles)} из {len(articles)} статей")
        return processed_articles
    
    def save_config(self, filepath: str):
        """Сохранение конфигурации"""
        config_dict = {
            'replace_numbers': self.config.replace_numbers,
            'replace_urls': self.config.replace_urls,
            'replace_emails': self.config.replace_emails,
            'replace_phones': self.config.replace_phones,
            'replace_dates': self.config.replace_dates,
            'replace_times': self.config.replace_times,
            'replace_currencies': self.config.replace_currencies,
            'normalize_punctuation': self.config.normalize_punctuation,
            'normalize_quotes': self.config.normalize_quotes,
            'normalize_dashes': self.config.normalize_dashes,
            'normalize_spaces': self.config.normalize_spaces,
            'expand_abbreviations': self.config.expand_abbreviations,
            'expand_contractions': self.config.expand_contractions,
            'to_lowercase': self.config.to_lowercase,
            'remove_extra_punctuation': self.config.remove_extra_punctuation,
            'preserve_sentence_structure': self.config.preserve_sentence_structure,
            'language': self.language,
            'tokens': self.tokens
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Конфигурация сохранена в {filepath}")
    
    @classmethod
    def load_config(cls, filepath: str) -> 'UniversalPreprocessor':
        """Загрузка конфигурации"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        config = PreprocessingConfig(**{k: v for k, v in config_dict.items() 
                                      if k not in ['language', 'tokens']})
        
        preprocessor = cls(config, config_dict.get('language', 'russian'))
        preprocessor.tokens = config_dict.get('tokens', preprocessor.tokens)
        
        return preprocessor

def main():
    """Пример использования UniversalPreprocessor"""
    # Создание конфигурации
    config = PreprocessingConfig(
        replace_numbers=True,
        replace_urls=True,
        replace_emails=True,
        replace_phones=True,
        replace_dates=True,
        replace_times=True,
        replace_currencies=True,
        normalize_punctuation=True,
        normalize_quotes=True,
        normalize_dashes=True,
        normalize_spaces=True,
        expand_abbreviations=True,
        expand_contractions=True,
        to_lowercase=False
    )
    
    # Создание предпроцессора
    preprocessor = UniversalPreprocessor(config, language='russian')
    
    # Пример текста
    sample_text = """
    Вчера, 15.12.2023, в 14:30 по адресу ул. Ленина, д. 10, г. Москва состоялось 
    заседание ООО "Рога и копыта". На встрече присутствовали д-р Иванов И.И., 
    проф. Петров П.П. и др. специалисты. Обсудили вопросы на сумму 1,500,000 руб.
    Контакты: +7 (495) 123-45-67, email@example.com, сайт: https://example.com
    """
    
    print("Исходный текст:")
    print(sample_text)
    print("\n" + "="*60 + "\n")
    
    # Предобработка
    processed_text = preprocessor.preprocess_text(sample_text)
    
    print("Обработанный текст:")
    print(processed_text)
    print("\n" + "="*60 + "\n")
    
    # Сохранение конфигурации
    preprocessor.save_config("preprocessing_config.json")
    
    # Пример с полной конфигурацией
    full_config = PreprocessingConfig(
        replace_numbers=True,
        replace_urls=True,
        replace_emails=True,
        replace_phones=True,
        replace_dates=True,
        replace_times=True,
        replace_currencies=True,
        normalize_punctuation=True,
        normalize_quotes=True,
        normalize_dashes=True,
        normalize_spaces=True,
        expand_abbreviations=True,
        expand_contractions=True,
        to_lowercase=True
    )
    
    full_preprocessor = UniversalPreprocessor(full_config, language='russian')
    full_processed = full_preprocessor.preprocess_text(sample_text)
    
    print("Полная обработка (с приведением к нижнему регистру):")
    print(full_processed)

if __name__ == "__main__":
    main()

