import os
os.environ['TCL_LIBRARY'] = "C:/Program Files/Python313/tcl/tcl8.6"
os.environ['TK_LIBRARY'] = "C:/Program Files/Python313/tcl/tk8.6"
import requests
from bs4 import BeautifulSoup
import json
import time
import re
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
import logging
from typing import List, Dict, Optional
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class KommersantParser:
    """Парсер для сайта Коммерсантъ"""
    
    def __init__(self, base_url="https://www.kommersant.ru", delay=1.0):
        self.base_url = base_url
        self.delay = delay  # Задержка между запросами
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def get_article_by_id(self, article_id: int) -> Optional[Dict]:
        """Получение статьи по ID"""
        url = f"{self.base_url}/doc/{article_id}"
        return self.parse_article(url)
    
    def parse_article_range(self, start_id: int, end_id: int, max_articles: int = None) -> List[Dict]:
        """Парсинг статей в диапазоне ID"""
        articles = []
        current_id = start_id
        processed_count = 0
        
        logger.info(f"Начинаем парсинг статей с ID {start_id} по {end_id}")
        
        while current_id <= end_id and (max_articles is None or processed_count < max_articles):
            try:
                logger.info(f"Парсим статью ID {current_id}")
                article = self.get_article_by_id(current_id)
                
                if article:
                    articles.append(article)
                    processed_count += 1
                    logger.info(f"Успешно спарсена статья {current_id}: {article['title'][:50]}...")
                else:
                    logger.warning(f"Статья с ID {current_id} не найдена или не удалось спарсить")
                
                current_id += 1
                time.sleep(self.delay)
                
            except Exception as e:
                logger.error(f"Ошибка при парсинге статьи {current_id}: {e}")
                current_id += 1
                continue
        
        logger.info(f"Парсинг завершен. Обработано {processed_count} статей")
        return articles
    
    def parse_recent_articles(self, max_articles: int = 100) -> List[Dict]:
        """Парсинг недавних статей (начиная с высоких ID)"""
        # Начинаем с текущего времени и идем назад
        current_time = datetime.now()
        # Примерный ID на основе текущей даты (примерно 8 миллионов)
        start_id = 8000000
        end_id = start_id - max_articles
        
        return self.parse_article_range(end_id, start_id, max_articles)
    
    def parse_article(self, url: str) -> Optional[Dict]:
        """Парсинг отдельной статьи"""
        try:
            logger.info(f"Парсим статью: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Извлечение заголовка
            title = self._extract_title(soup)
            if not title:
                logger.warning(f"Не удалось извлечь заголовок для {url}")
                return None
            
            # Извлечение текста статьи
            text = self._extract_text(soup)
            if not text or len(text.strip()) < 100:
                logger.warning(f"Недостаточно текста для {url}")
                return None
            
            # Извлечение даты
            date = self._extract_date(soup, url)
            
            # Извлечение категории
            category = self._extract_category(soup, url)
            
            # Извлечение тегов
            tags = self._extract_tags(soup)
            
            # Извлечение автора
            author = self._extract_author(soup)
            
            article_data = {
                'title': title.strip(),
                'text': text.strip(),
                'date': date,
                'url': url,
                'category': category,
                'tags': tags,
                'author': author,
                'source': 'kommersant.ru',
                'parsed_at': datetime.now().isoformat()
            }
            
            logger.info(f"Успешно спарсена статья: {title[:50]}...")
            return article_data
            
        except Exception as e:
            logger.error(f"Ошибка при парсинге статьи {url}: {e}")
            return None
    
    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Извлечение заголовка статьи"""
        # Различные селекторы для заголовка Коммерсанта
        title_selectors = [
            'h1.doc_header_title',
            'h1.doc_header__title',
            'h1[class*="title"]',
            'h1',
            '.doc_header_title',
            '.doc_header__title',
            '.article__title',
            '.article__header__title'
        ]
        
        for selector in title_selectors:
            title_elem = soup.select_one(selector)
            if title_elem and title_elem.get_text().strip():
                return title_elem.get_text().strip()
        
        return None
    
    def _extract_text(self, soup: BeautifulSoup) -> Optional[str]:
        """Извлечение основного текста статьи"""
        # Удаление ненужных элементов
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'advertisement', 'ad']):
            element.decompose()
        
        # Различные селекторы для текста статьи Коммерсанта
        text_selectors = [
            '.doc_text',
            '.doc_text__content',
            '.article__text',
            '.article__body',
            '.article__content',
            '.article__text__content',
            '[class*="doc_text"]',
            '.article__body__content'
        ]
        
        text_parts = []
        
        for selector in text_selectors:
            text_elem = soup.select_one(selector)
            if text_elem:
                # Извлечение всех параграфов
                paragraphs = text_elem.find_all(['p', 'div'], recursive=True)
                for p in paragraphs:
                    text = p.get_text().strip()
                    if text and len(text) > 20:  # Фильтрация коротких фрагментов
                        text_parts.append(text)
                break
        
        if not text_parts:
            # Fallback: поиск по всему документу
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text().strip()
                if text and len(text) > 50:
                    text_parts.append(text)
        
        return ' '.join(text_parts) if text_parts else None
    
    def _extract_date(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Извлечение даты публикации"""
        # Поиск даты в мета-тегах
        date_selectors = [
            'meta[property="article:published_time"]',
            'meta[name="pubdate"]',
            'meta[name="date"]',
            'time[datetime]',
            '.doc_header_date',
            '.doc_header__date',
            '.article__date',
            '.article__time'
        ]
        
        for selector in date_selectors:
            elem = soup.select_one(selector)
            if elem:
                date_str = elem.get('content') or elem.get('datetime') or elem.get_text()
                if date_str:
                    try:
                        # Попытка парсинга даты
                        parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        return parsed_date.isoformat()
                    except:
                        continue
        
        # Извлечение даты из URL (если есть)
        date_match = re.search(r'/(\d{8})/', url)
        if date_match:
            try:
                date_str = date_match.group(1)
                parsed_date = datetime.strptime(date_str, '%Y%m%d')
                return parsed_date.isoformat()
            except:
                pass
        
        return None
    
    def _extract_category(self, soup: BeautifulSoup, url: str) -> Optional[str]:
        """Извлечение категории статьи"""
        # Поиск категории в мета-тегах
        category_selectors = [
            'meta[property="article:section"]',
            'meta[name="category"]',
            '.doc_header_section',
            '.doc_header__section',
            '.breadcrumbs a',
            '.article__breadcrumbs a',
            '.article__category'
        ]
        
        for selector in category_selectors:
            elem = soup.select_one(selector)
            if elem:
                category = elem.get('content') or elem.get_text()
                if category and category.strip():
                    return category.strip()
        
        # Извлечение категории из URL
        url_parts = urlparse(url).path.split('/')
        for part in url_parts:
            if part in ['politics', 'economy', 'society', 'world', 'sport', 'culture', 'technology', 'news', 'business', 'finance']:
                return part
        
        return None
    
    def _extract_tags(self, soup: BeautifulSoup) -> List[str]:
        """Извлечение тегов статьи"""
        tags = []
        
        # Поиск тегов
        tag_selectors = [
            '.doc_tags a',
            '.doc_tags__item',
            '.article__tags a',
            '.tags a',
            '.article__keywords a',
            'meta[name="keywords"]'
        ]
        
        for selector in tag_selectors:
            if selector.startswith('meta'):
                elem = soup.select_one(selector)
                if elem:
                    content = elem.get('content', '')
                    if content:
                        tags.extend([tag.strip() for tag in content.split(',')])
            else:
                elems = soup.select(selector)
                for elem in elems:
                    tag = elem.get_text().strip()
                    if tag:
                        tags.append(tag)
        
        return list(set(tags))  # Удаление дубликатов
    
    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Извлечение автора статьи"""
        author_selectors = [
            '.doc_header_author',
            '.doc_header__author',
            '.article__author',
            '.author',
            'meta[name="author"]',
            'meta[property="article:author"]'
        ]
        
        for selector in author_selectors:
            elem = soup.select_one(selector)
            if elem:
                author = elem.get('content') or elem.get_text()
                if author and author.strip():
                    return author.strip()
        
        return None
    
    def save_articles(self, articles: List[Dict], filename: str = "kommersant_articles.jsonl"):
        """Сохранение статей в JSONL формате"""
        # Определяем путь относительно текущего файла
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')
        
        logger.info(f"Сохранено {len(articles)} статей в файл {filepath}")
        return filepath
    
    def get_statistics(self, articles: List[Dict]) -> Dict:
        """Получение статистики по статьям"""
        if not articles:
            return {}
        
        total_words = sum(len(article['text'].split()) for article in articles)
        categories = {}
        authors = {}
        
        for article in articles:
            category = article.get('category', 'Неизвестно')
            categories[category] = categories.get(category, 0) + 1
            
            author = article.get('author')
            if author:
                authors[author] = authors.get(author, 0) + 1
        
        return {
            'total_articles': len(articles),
            'total_words': total_words,
            'avg_words_per_article': total_words // len(articles),
            'categories': categories,
            'top_authors': dict(sorted(authors.items(), key=lambda x: x[1], reverse=True)[:10])
        }

def main():
    """Основная функция для запуска парсера"""
    parser = KommersantParser(delay=1.0)
    
    # Парсинг недавних статей
    logger.info("Начинаем парсинг Коммерсанта")
    
    # Парсим статьи с ID от 8060000 до 8069999 (примерно 1000 статей)
    articles = parser.parse_article_range(8060000, 8069999, max_articles=2000)
    
    if articles:
        # Сохранение результатов
        filepath = parser.save_articles(articles)
        
        # Статистика
        stats = parser.get_statistics(articles)
        
        logger.info(f"Парсинг завершен!")
        logger.info(f"Всего статей: {stats['total_articles']}")
        logger.info(f"Всего слов: {stats['total_words']:,}")
        logger.info(f"Среднее количество слов на статью: {stats['avg_words_per_article']:,}")
        
        logger.info(f"Статьи по категориям:")
        for category, count in stats['categories'].items():
            logger.info(f"  {category}: {count} статей")
        
        if stats['top_authors']:
            logger.info(f"Топ авторов:")
            for author, count in list(stats['top_authors'].items())[:5]:
                logger.info(f"  {author}: {count} статей")
        
        logger.info(f"Результаты сохранены в: {filepath}")
    else:
        logger.error("Не удалось спарсить ни одной статьи")

if __name__ == "__main__":
    main()

