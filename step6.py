#!/usr/bin/env python3
"""
Этап 6: Разработка веб-интерфейса для интерактивного анализа
Streamlit приложение для интерактивного анализа токенизации
"""
import os
import sys
import json
import logging
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import time
import io
from datetime import datetime

# Добавляем путь к модулям проекта
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Импортируем модули проекта
try:
    from text_cleaner import TextCleaner
    from tokenization_analysis import TokenizationAnalyzer
    from subword_models import SubwordModelTrainer
except ImportError as e:
    st.error(f"Ошибка импорта модулей: {e}")
    st.stop()

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Настройка страницы
st.set_page_config(
    page_title="Интерактивный анализ токенизации",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

def load_sample_datasets() -> Dict[str, str]:
    """Загрузка примеров датасетов"""
    sample_texts = {
        "Русский текст (Коммерсантъ)": """
        Девелоперской группе Дипломат рассказали о возведении жилого комплекса Дом Учителя на Парковом проспекте, 1. 
        Объект находится в активной стадии строительства. В одной локации располагается множество детских образовательных учреждений - школ и детских садов.
        """,
        "Английский текст": """
        The quick brown fox jumps over the lazy dog. This is a sample English text for tokenization analysis. 
        Natural language processing involves many complex tasks including tokenization, normalization, and analysis.
        """,
        "Технический текст": """
        API endpoint /api/v1/users возвращает JSON объект с полями: id, name, email, created_at. 
        Используйте POST запрос для создания нового пользователя. Статус коды: 200 OK, 400 Bad Request, 500 Internal Server Error.
        """,
        "Смешанный текст": """
        В 2023 году компания OpenAI выпустила GPT-4. The model shows significant improvements in reasoning capabilities. 
        Стоимость API составляет $0.03 за 1K токенов. Performance metrics: accuracy 95%, latency <200ms.
        """
    }
    return sample_texts

@st.cache_data
def analyze_text_tokenization(text: str, method: str, language: str = "ru") -> Dict[str, Any]:
    """Анализ токенизации текста"""
    try:
        analyzer = TokenizationAnalyzer()
        
        # Выбираем метод токенизации
        tokens = []
        if method == "naive":
            tokens = analyzer.naive_tokenization(text)
        elif method == "regex":
            tokens = analyzer.regex_tokenization(text)
        elif method == "nltk":
            tokens = analyzer.nltk_tokenization(text)
        elif method == "spacy":
            tokens = analyzer.spacy_tokenization(text)
        elif method == "razdel":
            tokens = analyzer.razdel_tokenization(text)
        else:
            tokens = analyzer.naive_tokenization(text)
        
        # Проверяем, что токенизация прошла успешно
        if not tokens:
            return {"error": f"Метод {method} не вернул токены"}
        
        # Анализируем результаты
        analysis = {
            "method": method,
            "total_tokens": len(tokens),
            "unique_tokens": len(set(tokens)),
            "avg_token_length": sum(len(token) for token in tokens) / len(tokens) if tokens else 0,
            "tokens": tokens[:100],  # Первые 100 токенов для отображения
            "token_lengths": [len(token) for token in tokens],
            "token_frequencies": pd.Series(tokens).value_counts().head(20).to_dict()
        }
        
        return analysis
        
    except ImportError as e:
        error_msg = f"Библиотека не установлена для метода {method}: {e}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Ошибка анализа токенизации методом {method}: {e}"
        logger.error(error_msg)
        return {"error": error_msg}

def create_token_length_chart(analysis: Dict[str, Any]) -> go.Figure:
    """Создание графика распределения длин токенов"""
    if "token_lengths" not in analysis:
        return go.Figure()
    
    lengths = analysis["token_lengths"]
    length_counts = pd.Series(lengths).value_counts().sort_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=length_counts.index,
        y=length_counts.values,
        name='Количество токенов',
        marker_color='lightblue'
    ))
    
    fig.update_layout(
        title=f'Распределение длин токенов (метод: {analysis["method"]})',
        xaxis_title='Длина токена',
        yaxis_title='Количество токенов',
        showlegend=False
    )
    
    return fig

def create_token_frequency_chart(analysis: Dict[str, Any]) -> go.Figure:
    """Создание графика частотности токенов"""
    if "token_frequencies" not in analysis:
        return go.Figure()
    
    frequencies = analysis["token_frequencies"]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(frequencies.keys()),
        y=list(frequencies.values()),
        name='Частота',
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title=f'Топ-20 самых частых токенов (метод: {analysis["method"]})',
        xaxis_title='Токен',
        yaxis_title='Частота',
        xaxis_tickangle=-45,
        showlegend=False
    )
    
    return fig

def create_comparison_chart(analyses: List[Dict[str, Any]]) -> go.Figure:
    """Создание графика сравнения методов"""
    if not analyses:
        return go.Figure()
    
    methods = [a["method"] for a in analyses]
    total_tokens = [a["total_tokens"] for a in analyses]
    unique_tokens = [a["unique_tokens"] for a in analyses]
    avg_lengths = [a["avg_token_length"] for a in analyses]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=methods,
        y=total_tokens,
        name='Всего токенов',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        x=methods,
        y=unique_tokens,
        name='Уникальных токенов',
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title='Сравнение методов токенизации',
        xaxis_title='Метод',
        yaxis_title='Количество токенов',
        barmode='group'
    )
    
    return fig

def generate_html_report(analyses: List[Dict[str, Any]], text: str) -> str:
    """Генерация HTML отчета"""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Отчет по анализу токенизации</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .section {{ margin-bottom: 30px; }}
            .method {{ background-color: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Отчет по анализу токенизации</h1>
            <p>Дата создания: {datetime.now().strftime('%d.%m.%Y %H:%M')}</p>
        </div>
        
        <div class="section">
            <h2>Исходный текст</h2>
            <p style="background-color: #f9f9f9; padding: 15px; border-left: 4px solid #007acc;">
                {text[:500]}{'...' if len(text) > 500 else ''}
            </p>
        </div>
        
        <div class="section">
            <h2>Результаты анализа</h2>
    """
    
    for analysis in analyses:
        html += f"""
            <div class="method">
                <h3>Метод: {analysis['method']}</h3>
                <table>
                    <tr><th>Метрика</th><th>Значение</th></tr>
                    <tr><td>Всего токенов</td><td>{analysis['total_tokens']}</td></tr>
                    <tr><td>Уникальных токенов</td><td>{analysis['unique_tokens']}</td></tr>
                    <tr><td>Средняя длина токена</td><td>{analysis['avg_token_length']:.2f}</td></tr>
                </table>
            </div>
        """
    
    html += """
        </div>
    </body>
    </html>
    """
    
    return html

def main():
    """Главная функция веб-интерфейса"""
    st.title("🔤 Интерактивный анализ токенизации")
    st.markdown("---")
    
    # Боковая панель для настроек
    with st.sidebar:
        st.header("⚙️ Настройки анализа")
        
        # Выбор источника данных
        data_source = st.radio(
            "Источник данных:",
            ["Пример текста", "Загрузка файла", "Ввод текста"]
        )
        
        # Выбор языка
        language = st.selectbox(
            "Язык текста:",
            ["ru", "en", "mixed"],
            format_func=lambda x: {"ru": "Русский", "en": "Английский", "mixed": "Смешанный"}[x]
        )
        
        # Выбор методов токенизации
        available_methods = ["naive", "regex", "nltk", "spacy", "razdel"]
        method_labels = {
            "naive": "Простая токенизация",
            "regex": "Регулярные выражения", 
            "nltk": "NLTK",
            "spacy": "spaCy",
            "razdel": "Razdel (русский)"
        }
        
        selected_methods = st.multiselect(
            "Методы токенизации:",
            available_methods,
            default=["naive", "nltk", "spacy"],
            format_func=lambda x: method_labels[x]
        )
    
    # Основная область
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Исходный текст")
        
        # Загрузка текста в зависимости от выбранного источника
        text = ""
        
        if data_source == "Пример текста":
            sample_texts = load_sample_datasets()
            selected_sample = st.selectbox("Выберите пример:", list(sample_texts.keys()))
            text = sample_texts[selected_sample]
            st.text_area("Текст:", text, height=150, key="sample_text")
            
        elif data_source == "Загрузка файла":
            uploaded_file = st.file_uploader(
                "Загрузите текстовый файл:",
                type=['txt', 'json', 'csv'],
                help="Поддерживаются файлы: .txt, .json, .csv"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.txt'):
                        text = str(uploaded_file.read(), "utf-8")
                    elif uploaded_file.name.endswith('.json'):
                        data = json.load(uploaded_file)
                        text = str(data)
                    elif uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                        text = df.to_string()
                    
                    st.text_area("Загруженный текст:", text[:1000] + "..." if len(text) > 1000 else text, height=150)
                except Exception as e:
                    st.error(f"Ошибка загрузки файла: {e}")
                    
        elif data_source == "Ввод текста":
            text = st.text_area("Введите текст для анализа:", height=150, key="input_text")
    
    with col2:
        st.header("🎯 Действия")
        
        if st.button("🚀 Запустить анализ", type="primary"):
            if not text.strip():
                st.error("Введите текст для анализа!")
            elif not selected_methods:
                st.error("Выберите хотя бы один метод токенизации!")
            else:
                # Запуск анализа
                with st.spinner("Выполняется анализ..."):
                    analyses = []
                    errors = []
                    
                    for method in selected_methods:
                        analysis = analyze_text_tokenization(text, method, language)
                        if "error" not in analysis:
                            analyses.append(analysis)
                        else:
                            errors.append(f"{method}: {analysis['error']}")
                    
                    # Показываем ошибки, если есть
                    if errors:
                        st.warning("⚠️ Некоторые методы не удалось выполнить:")
                        for error in errors:
                            st.write(f"• {error}")
                    
                    # Показываем успешные результаты
                    if analyses:
                        st.success(f"✅ Анализ завершен! Успешно обработано методов: {len(analyses)}")
                    else:
                        st.error("❌ Не удалось выполнить анализ ни одним методом. Проверьте установку библиотек.")
                
                # Сохраняем результаты в session state
                st.session_state['analyses'] = analyses
                st.session_state['text'] = text
    
    # Отображение результатов
    if 'analyses' in st.session_state and st.session_state['analyses']:
        st.markdown("---")
        st.header("📊 Результаты анализа")
        
        analyses = st.session_state['analyses']
        
        # Общая статистика
        st.subheader("📈 Общая статистика")
        
        stats_data = []
        for analysis in analyses:
            stats_data.append({
                'Метод': analysis['method'],
                'Всего токенов': analysis['total_tokens'],
                'Уникальных токенов': analysis['unique_tokens'],
                'Средняя длина': f"{analysis['avg_token_length']:.2f}",
                'Коэффициент уникальности': f"{analysis['unique_tokens']/analysis['total_tokens']:.3f}"
            })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # Графики
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📏 Распределение длин токенов")
            if len(analyses) == 1:
                fig_length = create_token_length_chart(analyses[0])
                st.plotly_chart(fig_length, use_container_width=True)
            else:
                fig_comparison = create_comparison_chart(analyses)
                st.plotly_chart(fig_comparison, use_container_width=True)
        
        with col2:
            st.subheader("🔤 Частотность токенов")
            if analyses:
                fig_freq = create_token_frequency_chart(analyses[0])
                st.plotly_chart(fig_freq, use_container_width=True)
        
        # Детальные результаты по методам
        st.subheader("🔍 Детальные результаты")
        
        for analysis in analyses:
            with st.expander(f"Метод: {analysis['method']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Первые 50 токенов:**")
                    tokens_display = analysis['tokens'][:50]
                    st.write(", ".join(tokens_display))
                
                with col2:
                    st.write("**Топ-10 частых токенов:**")
                    top_tokens = list(analysis['token_frequencies'].items())[:10]
                    for token, freq in top_tokens:
                        st.write(f"• {token}: {freq}")
        
        # Экспорт отчета
        st.markdown("---")
        st.subheader("📄 Экспорт отчета")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Скачать HTML отчет"):
                html_report = generate_html_report(analyses, st.session_state['text'])
                
                st.download_button(
                    label="💾 Скачать HTML",
                    data=html_report,
                    file_name=f"tokenization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
        
        with col2:
            if st.button("📊 Скачать данные CSV"):
                csv_data = stats_df.to_csv(index=False)
                
                st.download_button(
                    label="💾 Скачать CSV",
                    data=csv_data,
                    file_name=f"tokenization_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Информационная панель
    st.markdown("---")
    st.subheader("ℹ️ Информация")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Методы токенизации:**
        - **Naive**: Простое разделение по пробелам
        - **Regex**: Использование регулярных выражений
        - **NLTK**: Библиотека Natural Language Toolkit
        - **spaCy**: Современная библиотека NLP
        - **Razdel**: Специализированная библиотека для русского языка
        """)
    
    with col2:
        st.info("""
        **Метрики анализа:**
        - **Всего токенов**: Общее количество слов/токенов
        - **Уникальных токенов**: Количество различных токенов
        - **Средняя длина**: Средняя длина токена в символах
        - **Коэффициент уникальности**: Доля уникальных токенов
        """)
    
    with col3:
        st.info("""
        **Возможности:**
        - Интерактивный выбор параметров
        - Визуализация результатов
        - Сравнение методов
        - Экспорт отчетов
        - Поддержка разных языков
        
        **Установка библиотек:**
        - pip install streamlit pandas plotly
        - pip install nltk spacy razdel
        - python -m spacy download ru_core_news_sm
        """)

if __name__ == "__main__":
    main()