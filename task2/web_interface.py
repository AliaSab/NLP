"""
Веб-интерфейс для анализа векторных пространств.

Реализует:
- Интерактивную векторную арифметику
- Эксперименты с семантическим сходством
- Визуализацию семантических осей
- Генерацию динамического отчёта
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
import os
import glob
from typing import List, Dict, Tuple, Any, Optional
import logging
from gensim.models import Word2Vec, FastText, Doc2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorSpaceWebInterface:
    """Класс для веб-интерфейса анализа векторных пространств."""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.model_path = None
        
    def find_model_files(self) -> Dict[str, List[str]]:
        """Поиск файлов моделей в текущей директории."""
        model_files = {
            'Word2Vec': [],
            'FastText': [],
            'Doc2Vec': [],
            'GloVe': []
        }
        
        # Поиск файлов .model
        model_patterns = {
            'Word2Vec': ['*word2vec*.model', '*Word2Vec*.model'],
            'FastText': ['*fasttext*.model', '*FastText*.model'],
            'Doc2Vec': ['*doc2vec*.model', '*Doc2Vec*.model'],
            'GloVe': ['*glove*.model', '*GloVe*.model']
        }
        
        for model_type, patterns in model_patterns.items():
            for pattern in patterns:
                files = glob.glob(pattern)
                model_files[model_type].extend(files)
        
        # Удаление дубликатов и сортировка
        for model_type in model_files:
            model_files[model_type] = sorted(list(set(model_files[model_type])))
        
        return model_files
        
    def load_model(self, model_path: str, model_type: str) -> bool:
        """Загрузка модели."""
        try:
            if model_type == 'Word2Vec':
                self.model = Word2Vec.load(model_path)
            elif model_type == 'FastText':
                self.model = FastText.load(model_path)
            elif model_type == 'Doc2Vec':
                self.model = Doc2Vec.load(model_path)
            elif model_type == 'GloVe':
                # Попытка загрузки GloVe
                try:
                    from glove import Glove
                    self.model = Glove.load(model_path)
                except ImportError:
                    try:
                        from glove_python import Glove
                        self.model = Glove.load(model_path)
                    except ImportError:
                        st.error("GloVe не установлен")
                        return False
            else:
                return False
            
            self.model_type = model_type
            self.model_path = model_path
            return True
        except Exception as e:
            st.error(f"Ошибка загрузки модели: {e}")
            return False
    
    def vector_arithmetic_calculator(self) -> None:
        """Интерактивный калькулятор векторной арифметики."""
        st.header("🧮 Калькулятор векторной арифметики")
        
        # Отладочная информация
        st.write(f"**Отладка:** model = {self.model is not None}, model_type = {self.model_type}")
        
        if self.model is None:
            st.warning("Сначала загрузите модель")
            return
        
        # Интерфейс для ввода выражения
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            word1 = st.text_input("Слово 1", value="король", help="Первое слово в выражении", key="word1_input")
        with col2:
            word2 = st.text_input("Слово 2", value="мужчина", help="Второе слово в выражении", key="word2_input")
        with col3:
            word3 = st.text_input("Слово 3", value="женщина", help="Третье слово в выражении", key="word3_input")
        with col4:
            top_n = st.number_input("Топ-N результатов", min_value=1, max_value=20, value=10)
        
        if st.button("Вычислить", type="primary", key="vector_arithmetic_btn"):
            if all([word1, word2, word3]):
                try:
                    # Проверка наличия слов в модели
                    missing_words = []
                    if hasattr(self.model, 'wv'):
                        # Word2Vec, FastText, Doc2Vec
                        for word in [word1, word2, word3]:
                            if word not in self.model.wv:
                                missing_words.append(word)
                    elif hasattr(self.model, 'dictionary'):
                        # GloVe
                        for word in [word1, word2, word3]:
                            if word not in self.model.dictionary:
                                missing_words.append(word)
                    
                    if missing_words:
                        st.error(f"Слова не найдены в модели: {', '.join(missing_words)}")
                        return
                    
                    # Векторная арифметика: word2 - word1 + word3
                    if hasattr(self.model, 'wv'):
                        result_vector = self.model.wv[word2] - self.model.wv[word1] + self.model.wv[word3]
                        similar_words = self.model.wv.similar_by_vector(result_vector, topn=top_n)
                    elif hasattr(self.model, 'dictionary'):
                        result_vector = (self.model.word_vectors[self.model.dictionary[word2]] - 
                                       self.model.word_vectors[self.model.dictionary[word1]] + 
                                       self.model.word_vectors[self.model.dictionary[word3]])
                        similar_words = self.model.most_similar_by_vector(result_vector, topn=top_n)
                    
                    # Отображение результатов
                    st.subheader("Результат векторной арифметики")
                    st.write(f"**Выражение:** {word2} - {word1} + {word3}")
                    
                    # Таблица результатов
                    results_data = []
                    for i, (word, similarity) in enumerate(similar_words, 1):
                        results_data.append({
                            'Ранг': i,
                            'Слово': word,
                            'Косинусное сходство': f"{similarity:.4f}"
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Визуализация промежуточных векторов
                    self.visualize_vector_arithmetic(word1, word2, word3, similar_words[:5])
                    
                except Exception as e:
                    st.error(f"Ошибка при вычислении: {e}")
            else:
                st.warning("Заполните все поля")
    
    def visualize_vector_arithmetic(self, word1: str, word2: str, word3: str, 
                                  similar_words: List[Tuple[str, float]]) -> None:
        """Визуализация промежуточных векторов в векторной арифметике."""
        try:
            # Получение векторов
            vectors = {}
            if hasattr(self.model, 'wv'):
                # Word2Vec, FastText, Doc2Vec
                vectors = {
                    word1: self.model.wv[word1],
                    word2: self.model.wv[word2],
                    word3: self.model.wv[word3]
                }
                
                # Добавление векторов результатов
                for word, _ in similar_words:
                    vectors[word] = self.model.wv[word]
            elif hasattr(self.model, 'dictionary'):
                # GloVe
                vectors = {
                    word1: self.model.word_vectors[self.model.dictionary[word1]],
                    word2: self.model.word_vectors[self.model.dictionary[word2]],
                    word3: self.model.word_vectors[self.model.dictionary[word3]]
                }
                
                # Добавление векторов результатов
                for word, _ in similar_words:
                    if word in self.model.dictionary:
                        vectors[word] = self.model.word_vectors[self.model.dictionary[word]]
            
            # Снижение размерности для визуализации
            words = list(vectors.keys())
            vectors_array = np.array(list(vectors.values()))
            
            # PCA для снижения размерности
            pca = PCA(n_components=2, random_state=42)
            vectors_2d = pca.fit_transform(vectors_array)
            
            # Создание scatter plot
            fig = go.Figure()
            
            # Цвета для разных типов слов
            colors = {
                word1: 'red',
                word2: 'blue', 
                word3: 'green'
            }
            
            for i, word in enumerate(words):
                color = colors.get(word, 'gray')
                size = 15 if word in [word1, word2, word3] else 10
                
                fig.add_trace(go.Scatter(
                    x=[vectors_2d[i, 0]],
                    y=[vectors_2d[i, 1]],
                    mode='markers+text',
                    marker=dict(size=size, color=color),
                    text=word,
                    textposition="top center",
                    name=word,
                    showlegend=False
                ))
            
            fig.update_layout(
                title="Визуализация векторной арифметики",
                xaxis_title="Первая компонента PCA",
                yaxis_title="Вторая компонента PCA",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Не удалось создать визуализацию: {e}")
    
    def semantic_similarity_calculator(self) -> None:
        """Калькулятор семантического сходства."""
        st.header("📏 Калькулятор семантического сходства")
        
        if self.model is None:
            st.warning("Сначала загрузите модель")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            word1 = st.text_input("Первое слово", value="дом", key="similarity_word1")
        with col2:
            word2 = st.text_input("Второе слово", value="жилище", key="similarity_word2")
        
        if st.button("Вычислить сходство", type="primary", key="similarity_btn"):
            if word1 and word2:
                try:
                    # Проверка наличия слов в модели
                    if hasattr(self.model, 'wv'):
                        # Word2Vec, FastText, Doc2Vec
                        if word1 not in self.model.wv or word2 not in self.model.wv:
                            st.error("Одно или оба слова не найдены в модели")
                            return
                        similarity = self.model.wv.similarity(word1, word2)
                    elif hasattr(self.model, 'dictionary'):
                        # GloVe
                        if word1 not in self.model.dictionary or word2 not in self.model.dictionary:
                            st.error("Одно или оба слова не найдены в модели")
                        return
                        similarity = self.model.similarity(word1, word2)
                    else:
                        st.error("Неподдерживаемый тип модели")
                        return
                    
                    # Отображение результата
                    st.metric("Косинусное сходство", f"{similarity:.4f}")
                    
                    # Интерпретация сходства
                    if similarity > 0.8:
                        interpretation = "Очень высокое сходство"
                        color = "green"
                    elif similarity > 0.6:
                        interpretation = "Высокое сходство"
                        color = "lightgreen"
                    elif similarity > 0.4:
                        interpretation = "Умеренное сходство"
                        color = "yellow"
                    elif similarity > 0.2:
                        interpretation = "Низкое сходство"
                        color = "orange"
                    else:
                        interpretation = "Очень низкое сходство"
                        color = "red"
                    
                    st.markdown(f"**Интерпретация:** <span style='color: {color}'>{interpretation}</span>", 
                              unsafe_allow_html=True)
                    
                    # График сходства
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = similarity,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Семантическое сходство"},
                        delta = {'reference': 0.5},
                        gauge = {
                            'axis': {'range': [None, 1]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 0.2], 'color': "lightgray"},
                                {'range': [0.2, 0.4], 'color': "yellow"},
                                {'range': [0.4, 0.6], 'color': "orange"},
                                {'range': [0.6, 0.8], 'color': "lightgreen"},
                                {'range': [0.8, 1], 'color': "green"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 0.5
                            }
                        }
                    ))
                    
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Ошибка при вычислении сходства: {e}")
            else:
                st.warning("Заполните оба поля")
    
    def semantic_axis_analyzer(self) -> None:
        """Анализатор семантических осей."""
        st.header("📊 Анализатор семантических осей")
        
        if self.model is None:
            st.warning("Сначала загрузите модель")
            return
        
        # Предустановленные оси
        preset_axes = {
            "Мужчина-Женщина": ("мужчина", "женщина"),
            "Большой-Маленький": ("большой", "маленький"),
            "Хороший-Плохой": ("хороший", "плохой"),
            "Горячий-Холодный": ("горячий", "холодный"),
            "Быстрый-Медленный": ("быстрый", "медленный")
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            axis_choice = st.selectbox("Выберите ось", ["Пользовательская"] + list(preset_axes.keys()), key="axis_choice")
        
        if axis_choice == "Пользовательская":
            with col2:
                word1 = st.text_input("Первое слово оси", value="мужчина", key="axis_word1")
                word2 = st.text_input("Второе слово оси", value="женщина", key="axis_word2")
        else:
            word1, word2 = preset_axes[axis_choice]
            with col2:
                st.info(f"Ось: {word1} ↔ {word2}")
        
        if st.button("Анализировать ось", type="primary", key="axis_analysis_btn"):
            if word1 and word2:
                try:
                    # Проверка наличия слов в модели
                    if hasattr(self.model, 'wv'):
                        # Word2Vec, FastText, Doc2Vec
                        if word1 not in self.model.wv or word2 not in self.model.wv:
                            st.error("Одно или оба слова не найдены в модели")
                            return
                        
                        # Вычисление оси
                        axis_vector = self.model.wv[word2] - self.model.wv[word1]
                        axis_vector = axis_vector / np.linalg.norm(axis_vector)
                        
                        # Проекция слов на ось
                        vocab = list(self.model.wv.key_to_index.keys())
                        projections = {}
                        
                        # Ограничиваем для производительности
                        sample_size = min(1000, len(vocab))
                        sample_words = np.random.choice(vocab, sample_size, replace=False)
                        
                        for word in sample_words:
                            if word in self.model.wv:
                                projection = np.dot(self.model.wv[word], axis_vector)
                                projections[word] = projection
                                
                    elif hasattr(self.model, 'dictionary'):
                        # GloVe
                        if word1 not in self.model.dictionary or word2 not in self.model.dictionary:
                            st.error("Одно или оба слова не найдены в модели")
                            return
                        
                        # Вычисление оси
                        axis_vector = (self.model.word_vectors[self.model.dictionary[word2]] - 
                                     self.model.word_vectors[self.model.dictionary[word1]])
                        axis_vector = axis_vector / np.linalg.norm(axis_vector)
                        
                        # Проекция слов на ось
                        projections = {}
                        
                        # Ограничиваем для производительности
                        sample_size = min(1000, len(self.model.dictionary))
                        sample_words = np.random.choice(list(self.model.dictionary.keys()), 
                                                      sample_size, replace=False)
                        
                        for word in sample_words:
                            if word in self.model.dictionary:
                                projection = np.dot(self.model.word_vectors[self.model.dictionary[word]], axis_vector)
                                projections[word] = projection
                    else:
                        st.error("Неподдерживаемый тип модели")
                        return
                    
                    # Сортировка по проекции
                    sorted_projections = sorted(projections.items(), key=lambda x: x[1])
                    
                    # Отображение результатов
                    st.subheader(f"Анализ оси: {word1} ↔ {word2}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Наиболее отрицательные проекции:**")
                        negative_words = sorted_projections[:10]
                        for word, proj in negative_words:
                            st.write(f"• {word}: {proj:.4f}")
                    
                    with col2:
                        st.write("**Наиболее положительные проекции:**")
                        positive_words = sorted_projections[-10:]
                        for word, proj in reversed(positive_words):
                            st.write(f"• {word}: {proj:.4f}")
                    
                    # Визуализация распределения проекций
                    projections_values = list(projections.values())
                    
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=projections_values,
                        nbinsx=50,
                        name="Распределение проекций"
                    ))
                    
                    fig.update_layout(
                        title="Распределение проекций слов на семантическую ось",
                        xaxis_title="Проекция на ось",
                        yaxis_title="Количество слов",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Ошибка при анализе оси: {e}")
            else:
                st.warning("Заполните оба поля")
    
    def nearest_neighbors_explorer(self) -> None:
        """Исследователь ближайших соседей."""
        st.header("🔍 Исследователь ближайших соседей")
        
        if self.model is None:
            st.warning("Сначала загрузите модель")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            word = st.text_input("Слово для поиска соседей", value="дом", key="neighbors_word")
        with col2:
            top_n = st.number_input("Количество соседей", min_value=1, max_value=50, value=10, key="neighbors_count")
        
        if st.button("Найти соседей", type="primary", key="neighbors_btn"):
            if word:
                try:
                    # Проверка наличия слова в модели
                    if hasattr(self.model, 'wv'):
                        # Word2Vec, FastText, Doc2Vec
                        if word not in self.model.wv:
                            st.error(f"Слово '{word}' не найдено в модели")
                            return
                        
                        # Поиск ближайших соседей
                        similar_words = self.model.wv.most_similar(word, topn=top_n)
                        
                    elif hasattr(self.model, 'dictionary'):
                        # GloVe
                        if word not in self.model.dictionary:
                            st.error(f"Слово '{word}' не найдено в модели")
                            return
                        
                        # Поиск ближайших соседей
                        similar_words = self.model.most_similar(word, topn=top_n)
                    else:
                        st.error("Неподдерживаемый тип модели")
                        return
                    
                    # Отображение результатов
                    st.subheader(f"Ближайшие соседи для слова '{word}'")
                    
                    # Таблица результатов
                    neighbors_data = []
                    for i, (neighbor, similarity) in enumerate(similar_words, 1):
                        neighbors_data.append({
                            'Ранг': i,
                            'Слово': neighbor,
                            'Косинусное сходство': f"{similarity:.4f}"
                        })
                    
                    neighbors_df = pd.DataFrame(neighbors_data)
                    st.dataframe(neighbors_df, use_container_width=True)
                    
                    # Визуализация сходства
                    fig = go.Figure(go.Bar(
                        x=[neighbor for neighbor, _ in similar_words],
                        y=[similarity for _, similarity in similar_words],
                        marker_color='lightblue'
                    ))
                    
                    fig.update_layout(
                        title=f"Косинусное сходство с соседями слова '{word}'",
                        xaxis_title="Соседние слова",
                        yaxis_title="Косинусное сходство",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Ошибка при поиске соседей: {e}")
            else:
                st.warning("Введите слово")
    
    def generate_report(self) -> None:
        """Генерация динамического отчёта."""
        st.header("📋 Динамический отчёт")
        
        if self.model is None:
            st.warning("Сначала загрузите модель")
            return
        
        if st.button("Сгенерировать отчёт", type="primary", key="report_btn"):
            with st.spinner("Генерация отчёта..."):
                try:
                    # Информация о модели
                    st.subheader("Информация о модели")
                    
                    model_info = {
                        "Тип модели": self.model_type,
                        "Путь к модели": self.model_path,
                        "Размер словаря": len(self.model.wv),
                        "Размерность векторов": self.model.wv.vector_size
                    }
                    
                    for key, value in model_info.items():
                        st.metric(key, value)
                    
                    # Статистика по словарю
                    st.subheader("Статистика словаря")
                    
                    vocab_size = len(self.model.wv)
                    st.write(f"Общий размер словаря: {vocab_size:,} слов")
                    
                    # Анализ длины слов
                    word_lengths = [len(word) for word in self.model.wv.key_to_index.keys()]
                    avg_length = np.mean(word_lengths)
                    st.write(f"Средняя длина слова: {avg_length:.2f} символов")
                    
                    # Топ-10 самых частых слов (если доступно)
                    if hasattr(self.model, 'wv') and hasattr(self.model.wv, 'index_to_key'):
                        st.subheader("Примеры слов из словаря")
                        sample_words = list(self.model.wv.key_to_index.keys())[:20]
                        st.write(", ".join(sample_words))
                    
                    # Визуализация распределения длин слов
                    fig = go.Figure(go.Histogram(
                        x=word_lengths,
                        nbinsx=20,
                        name="Распределение длин слов"
                    ))
                    
                    fig.update_layout(
                        title="Распределение длин слов в словаре",
                        xaxis_title="Длина слова (символы)",
                        yaxis_title="Количество слов",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.success("Отчёт сгенерирован успешно!")
                    
                except Exception as e:
                    st.error(f"Ошибка при генерации отчёта: {e}")


def main():
    """Основная функция веб-интерфейса."""
    st.set_page_config(
        page_title="Анализ векторных пространств",
        page_icon="🧠",
        layout="wide"
    )
    
    st.title("🧠 Анализ векторных пространств")
    st.markdown("Интерактивный инструмент для исследования семантических представлений слов")
    
    # Создание экземпляра интерфейса
    interface = VectorSpaceWebInterface()
    
    # Инициализация состояния сессии
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
    if 'current_model_path' not in st.session_state:
        st.session_state.current_model_path = None
    if 'current_model_type' not in st.session_state:
        st.session_state.current_model_type = None
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    
    # Восстановление модели из состояния сессии
    if st.session_state.model_loaded and st.session_state.current_model is not None:
        interface.model = st.session_state.current_model
        interface.model_type = st.session_state.current_model_type
        interface.model_path = st.session_state.current_model_path
    
    # Боковая панель для загрузки модели
    with st.sidebar:
        st.header("📁 Загрузка модели")
        
        # Поиск доступных моделей
        available_models = interface.find_model_files()
        
        # Выбор типа модели
        model_type = st.selectbox(
            "Тип модели",
            ["Word2Vec", "FastText", "Doc2Vec", "GloVe"],
            key="model_type_select"
        )
        
        # Выбор модели из найденных файлов
        if available_models[model_type]:
            st.write(f"**Найденные модели {model_type}:**")
            selected_model = st.selectbox(
                f"Выберите модель {model_type}",
                ["Введите путь вручную"] + available_models[model_type],
                key=f"model_select_{model_type}"
            )
            
            if selected_model != "Введите путь вручную":
                model_path = selected_model
                st.success(f"Выбрана модель: {model_path}")
            else:
                model_path = st.text_input(
                    "Путь к модели",
                    value="word2vec_skip-gram_size200_win5_min5.model",
                    help="Введите путь к файлу модели",
                    key="manual_model_path"
                )
        else:
            st.warning(f"Модели {model_type} не найдены в текущей директории")
        model_path = st.text_input(
            "Путь к модели",
            value="word2vec_skip-gram_size200_win5_min5.model",
                help="Введите путь к файлу модели",
                key="manual_model_path_fallback"
        )
        
        # Кнопка загрузки модели
        if st.button("Загрузить модель", type="primary", key="load_model_btn"):
            if model_path and os.path.exists(model_path):
                with st.spinner("Загрузка модели..."):
                    if interface.load_model(model_path, model_type):
                        st.session_state.model_loaded = True
                        st.session_state.current_model_path = model_path
                        st.session_state.current_model_type = model_type
                        st.session_state.current_model = interface.model
                        st.success("Модель загружена успешно!")
                        st.rerun()
                    else:
                        st.error("Ошибка загрузки модели")
            else:
                st.error("Файл модели не найден")
        
        # Информация о загруженной модели
        if st.session_state.model_loaded:
            st.header("ℹ️ Информация о модели")
            st.write(f"**Тип:** {st.session_state.current_model_type}")
            
            # Определение размера словаря и размерности
            if hasattr(interface.model, 'wv'):
                vocab_size = len(interface.model.wv)
                vector_size = interface.model.wv.vector_size
            elif hasattr(interface.model, 'dictionary'):
                vocab_size = len(interface.model.dictionary)
                vector_size = interface.model.word_vectors.shape[1]
            else:
                vocab_size = "неизвестно"
                vector_size = "неизвестно"
            
            st.write(f"**Словарь:** {vocab_size:,} слов")
            st.write(f"**Размерность:** {vector_size}")
            st.write(f"**Путь:** {st.session_state.current_model_path}")
            
            # Кнопка сброса модели
            if st.button("Сбросить модель", type="secondary", key="reset_model_btn"):
                st.session_state.model_loaded = False
                st.session_state.current_model_path = None
                st.session_state.current_model_type = None
                st.session_state.current_model = None
                interface.model = None
                st.rerun()
    
    # Отладочная информация
    st.write(f"**Отладка сессии:** model_loaded = {st.session_state.model_loaded}, model = {interface.model is not None}")
    
    # Основной интерфейс
    if st.session_state.model_loaded:
        # Создание вкладок
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🧮 Векторная арифметика",
            "📏 Семантическое сходство", 
            "📊 Семантические оси",
            "🔍 Ближайшие соседи",
            "📋 Отчёт"
        ])
        
        with tab1:
            interface.vector_arithmetic_calculator()
        
        with tab2:
            interface.semantic_similarity_calculator()
        
        with tab3:
            interface.semantic_axis_analyzer()
        
        with tab4:
            interface.nearest_neighbors_explorer()
        
        with tab5:
            interface.generate_report()
    
    else:
        st.info("👈 Загрузите модель в боковой панели для начала работы")
        
        # Демонстрационные данные
        st.header("📚 О программе")
        st.markdown("""
        Этот инструмент позволяет:
        
        - **Векторная арифметика**: Выполнять операции типа "король - мужчина + женщина = королева"
        - **Семантическое сходство**: Измерять косинусное сходство между словами
        - **Семантические оси**: Анализировать направления в векторном пространстве
        - **Ближайшие соседи**: Находить семантически близкие слова
        - **Динамические отчёты**: Генерировать статистику по модели
        
        Для начала работы загрузите обученную модель в боковой панели.
        """)
        
        # Показ доступных моделей
        st.header("📂 Доступные модели")
        available_models = interface.find_model_files()
        
        for model_type, files in available_models.items():
            if files:
                st.write(f"**{model_type}:**")
                for file in files:
                    st.write(f"• {file}")
            else:
                st.write(f"**{model_type}:** Модели не найдены")


if __name__ == "__main__":
    main()

