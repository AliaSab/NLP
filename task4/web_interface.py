"""
Веб-интерфейс для кластеризации текстов.
Использует Streamlit.
"""

import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Добавляем пути
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Task1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Task2'))

from text_preprocessing import TextPreprocessor, load_articles_from_jsonl, preprocess_articles
from text_to_vector import TextVectorizer, find_embedding_models
from clustering import ClusteringPipeline, tune_kmeans, tune_dbscan, tune_hierarchical, tune_gmm, tune_spectral
from evaluation import ClusteringEvaluator
from interpretation import ClusteringInterpreter

# Настройка страницы
st.set_page_config(
    page_title="Кластеризация текстов",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Кластеризация текстов")
st.markdown("---")

# Инициализация состояния сессии
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'clustering' not in st.session_state:
    st.session_state.clustering = None
if 'articles' not in st.session_state:
    st.session_state.articles = []
if 'tokenized_texts' not in st.session_state:
    st.session_state.tokenized_texts = []
if 'vectors' not in st.session_state:
    st.session_state.vectors = None
if 'labels' not in st.session_state:
    st.session_state.labels = None
if 'clustering_method' not in st.session_state:
    st.session_state.clustering_method = None

# Боковая панель
with st.sidebar:
    st.header("⚙️ Настройки")
    
    # Загрузка данных
    st.subheader("1. Загрузка данных")
    data_file = st.file_uploader(
        "Загрузите JSONL файл",
        type=['jsonl'],
        help="Файл kommersant_articles_processed.jsonl из Task1"
    )
    
    if data_file is not None:
        if st.button("Загрузить данные"):
            with st.spinner("Загрузка данных..."):
                # Сохраняем временный файл
                temp_path = f"temp_{data_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(data_file.getbuffer())
                
                # Загружаем статьи
                articles = load_articles_from_jsonl(temp_path)
                st.session_state.articles = articles
                
                # Удаляем временный файл
                os.remove(temp_path)
                
                st.success(f"Загружено {len(articles)} статей")
    
    if st.session_state.articles:
        st.info(f"📄 Загружено статей: {len(st.session_state.articles)}")
    
    st.markdown("---")
    
    # Предобработка
    st.subheader("2. Предобработка")
    tokenization_method = st.selectbox(
        "Метод токенизации",
        ['whitespace', 'regex', 'bpe', 'sentencepiece']
    )
    
    use_lemmatization = st.checkbox("Лемматизация", value=True)
    
    vocab_size = None
    if tokenization_method in ['bpe', 'sentencepiece']:
        vocab_size = st.selectbox(
            "Размер словаря",
            [8000, 16000, 32000],
            index=0
        )
    
    if st.button("Предобработать"):
        if not st.session_state.articles:
            st.error("Сначала загрузите данные!")
        else:
            with st.spinner("Предобработка..."):
                preprocessor = TextPreprocessor()
                
                # Загрузка BPE/SentencePiece моделей
                task1_path = os.path.join(os.path.dirname(__file__), '..', 'Task1')
                if tokenization_method == 'bpe':
                    bpe_path = os.path.join(task1_path, f'bpe_model_{vocab_size}.json')
                    if os.path.exists(bpe_path):
                        preprocessor.load_bpe_model(bpe_path, vocab_size)
                elif tokenization_method == 'sentencepiece':
                    sp_path = os.path.join(task1_path, f'sp_bpe_{vocab_size}.model')
                    if os.path.exists(sp_path):
                        preprocessor.load_sentencepiece_model(sp_path, vocab_size)
                
                st.session_state.preprocessor = preprocessor
                
                # Предобработка
                tokenized = preprocess_articles(
                    st.session_state.articles,
                    preprocessor,
                    tokenization_method=tokenization_method,
                    lemmatize=use_lemmatization,
                    vocab_size=vocab_size
                )
                st.session_state.tokenized_texts = tokenized
                
                st.success(f"Предобработано {len(tokenized)} статей")
    
    st.markdown("---")
    
    # Векторизация
    st.subheader("3. Векторизация")
    vectorization_method = st.selectbox(
        "Метод векторизации",
        ['tfidf', 'bm25', 'word2vec', 'fasttext', 'glove']
    )
    
    if vectorization_method in ['word2vec', 'fasttext', 'glove']:
        task2_path = os.path.join(os.path.dirname(__file__), '..', 'Task2')
        models = find_embedding_models(task2_path)
        
        model_type = vectorization_method
        if models[model_type]:
            model_files = [os.path.basename(p) for p in models[model_type]]
            selected_model = st.selectbox(
                f"Выберите модель {model_type}",
                model_files
            )
            model_path = os.path.join(task2_path, selected_model)
        else:
            st.warning(f"Модели {model_type} не найдены в Task2")
            model_path = None
    else:
        model_path = None
    
    if st.button("Векторизовать"):
        if not st.session_state.tokenized_texts:
            st.error("Сначала выполните предобработку!")
        else:
            with st.spinner("Векторизация..."):
                vectorizer = TextVectorizer()
                st.session_state.vectorizer = vectorizer
                
                if vectorization_method == 'tfidf':
                    vectorizer.fit_tfidf(st.session_state.tokenized_texts)
                    vectors = vectorizer.transform_tfidf(st.session_state.tokenized_texts)
                elif vectorization_method == 'bm25':
                    vectorizer.fit_bm25(st.session_state.tokenized_texts)
                    vectors = vectorizer.transform_bm25(st.session_state.tokenized_texts)
                elif vectorization_method in ['word2vec', 'fasttext', 'glove']:
                    if model_path and os.path.exists(model_path):
                        try:
                            if vectorization_method == 'word2vec':
                                if vectorizer.load_word2vec(model_path):
                                    vectors = vectorizer.transform_embeddings(
                                        st.session_state.tokenized_texts,
                                        model_type=vectorization_method
                                    )
                                else:
                                    st.error("Не удалось загрузить модель Word2Vec")
                                    vectors = None
                            elif vectorization_method == 'fasttext':
                                if vectorizer.load_fasttext(model_path):
                                    vectors = vectorizer.transform_embeddings(
                                        st.session_state.tokenized_texts,
                                        model_type=vectorization_method
                                    )
                                else:
                                    st.error("Не удалось загрузить модель FastText")
                                    vectors = None
                            elif vectorization_method == 'glove':
                                if vectorizer.load_glove(model_path):
                                    vectors = vectorizer.transform_embeddings(
                                        st.session_state.tokenized_texts,
                                        model_type=vectorization_method
                                    )
                                else:
                                    st.error("Не удалось загрузить модель GloVe")
                                    vectors = None
                        except Exception as e:
                            st.error(f"Ошибка при загрузке модели: {e}")
                            vectors = None
                    else:
                        st.error("Модель не найдена!")
                        vectors = None
                else:
                    vectors = None
                
                if vectors is not None:
                    st.session_state.vectors = vectors
                    st.success(f"Векторизация завершена. Размерность: {vectors.shape}")
    
    st.markdown("---")
    
    # Кластеризация
    st.subheader("4. Кластеризация")
    clustering_method = st.selectbox(
        "Метод кластеризации",
        ['kmeans', 'dbscan', 'hierarchical', 'gmm', 'spectral']
    )
    
    if clustering_method == 'kmeans':
        n_clusters = st.slider("Количество кластеров", 2, 20, 5)
    elif clustering_method == 'dbscan':
        eps = st.slider("eps", 0.1, 1.0, 0.5, 0.1)
        min_samples = st.slider("min_samples", 3, 20, 5)
    elif clustering_method == 'hierarchical':
        n_clusters = st.slider("Количество кластеров", 2, 20, 5)
        linkage = st.selectbox("linkage", ['ward', 'complete', 'average'])
    elif clustering_method == 'gmm':
        n_components = st.slider("Количество компонент", 2, 20, 5)
    elif clustering_method == 'spectral':
        n_clusters = st.slider("Количество кластеров", 2, 20, 5)
    
    if st.button("Кластеризовать"):
        if st.session_state.vectors is None:
            st.error("Сначала выполните векторизацию!")
        else:
            with st.spinner("Кластеризация..."):
                clustering = ClusteringPipeline()
                st.session_state.clustering = clustering
                
                if clustering_method == 'kmeans':
                    labels = clustering.kmeans(st.session_state.vectors, n_clusters=n_clusters)
                elif clustering_method == 'dbscan':
                    labels = clustering.dbscan(st.session_state.vectors, eps=eps, min_samples=min_samples)
                elif clustering_method == 'hierarchical':
                    labels = clustering.hierarchical(
                        st.session_state.vectors,
                        n_clusters=n_clusters,
                        linkage=linkage
                    )
                elif clustering_method == 'gmm':
                    labels = clustering.gmm(st.session_state.vectors, n_components=n_components)
                elif clustering_method == 'spectral':
                    labels = clustering.spectral(st.session_state.vectors, n_clusters=n_clusters)
                else:
                    labels = None
                
                if labels is not None:
                    st.session_state.labels = labels
                    st.session_state.clustering_method = clustering_method
                    n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
                    st.success(f"Кластеризация завершена. Найдено кластеров: {n_clusters_found}")

# Основная область
if st.session_state.articles:
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Результаты", "📈 Метрики", "🔍 Интерпретация", "📋 Данные"])
    
    with tab1:
        st.header("Результаты кластеризации")
        
        if st.session_state.labels is not None:
            # Статистика по кластерам
            unique_labels, counts = np.unique(st.session_state.labels, return_counts=True)
            cluster_stats = pd.DataFrame({
                'Кластер': unique_labels,
                'Количество документов': counts
            })
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Распределение по кластерам")
                st.dataframe(cluster_stats, use_container_width=True)
            
            with col2:
                st.subheader("Визуализация распределения")
                fig = px.bar(cluster_stats, x='Кластер', y='Количество документов',
                           title='Размеры кластеров')
                st.plotly_chart(fig, use_container_width=True)
            
            # Визуализация кластеров
            if st.session_state.vectors is not None:
                st.subheader("Визуализация кластеров")
                
                reduction_method = st.selectbox("Метод снижения размерности", ['umap', 'pca'])
                
                if st.button("Построить визуализацию"):
                    with st.spinner("Построение визуализации..."):
                        if reduction_method == 'umap':
                            reduced = st.session_state.clustering.reduce_dimensions(
                                st.session_state.vectors, method='umap'
                            )
                        else:
                            reduced = st.session_state.clustering.reduce_dimensions(
                                st.session_state.vectors, method='pca'
                            )
                        
                        # Создаем DataFrame для Plotly
                        df_viz = pd.DataFrame({
                            'x': reduced[:, 0],
                            'y': reduced[:, 1],
                            'Кластер': st.session_state.labels
                        })
                        
                        fig = px.scatter(df_viz, x='x', y='y', color='Кластер',
                                       title=f'Визуализация кластеров ({reduction_method.upper()})',
                                       labels={'x': f'{reduction_method.upper()} 1',
                                              'y': f'{reduction_method.upper()} 2'})
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Выполните кластеризацию для просмотра результатов")
    
    with tab2:
        st.header("Метрики качества")
        
        if st.session_state.labels is not None and st.session_state.vectors is not None:
            evaluator = ClusteringEvaluator()
            metrics = evaluator.compute_internal_metrics(
                st.session_state.vectors,
                st.session_state.labels
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Silhouette Score", f"{metrics['silhouette']:.4f}",
                         help="Чем выше, тем лучше (диапазон [-1, 1])")
            with col2:
                st.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz']:.2f}",
                         help="Чем выше, тем лучше")
            with col3:
                st.metric("Davies-Bouldin", f"{metrics['davies_bouldin']:.4f}",
                         help="Чем ниже, тем лучше")
            
            # Подбор параметров для выбранного метода кластеризации
            if st.session_state.clustering_method:
                method_name = st.session_state.clustering_method
                st.subheader(f"Подбор параметров {method_name.upper()}")
                
                if method_name == 'kmeans':
                    if st.button("Подобрать оптимальное k"):
                        with st.spinner("Подбор параметров..."):
                            k_range = range(2, min(11, len(st.session_state.articles) // 10 + 1))
                            results = tune_kmeans(st.session_state.vectors, k_range=k_range)
                            
                            # Создаем DataFrame
                            df_tune = pd.DataFrame(results).T
                            df_tune.index.name = 'k'
                            df_tune = df_tune.reset_index()
                            
                            st.dataframe(df_tune, use_container_width=True)
                            
                            # График
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_tune['k'], y=df_tune['silhouette'],
                                                   mode='lines+markers', name='Silhouette'))
                            fig.update_layout(title='Зависимость Silhouette Score от k',
                                            xaxis_title='k', yaxis_title='Silhouette Score')
                            st.plotly_chart(fig, use_container_width=True)
                
                elif method_name == 'dbscan':
                    if st.button("Подобрать оптимальные параметры"):
                        with st.spinner("Подбор параметров..."):
                            eps_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                            min_samples_range = [3, 5, 7, 10]
                            results = tune_dbscan(st.session_state.vectors, 
                                                 eps_range=eps_range, 
                                                 min_samples_range=min_samples_range)
                            
                            # Создаем DataFrame
                            df_tune = pd.DataFrame(results).T
                            df_tune = df_tune.reset_index()
                            df_tune = df_tune.sort_values('silhouette', ascending=False)
                            
                            st.dataframe(df_tune, use_container_width=True)
                            
                            # График зависимости от eps
                            eps_values = [r['eps'] for r in results.values()]
                            silhouette_values = [r['silhouette'] for r in results.values()]
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=eps_values, y=silhouette_values,
                                                   mode='markers', name='Silhouette',
                                                   text=[f"min_samples={r['min_samples']}" for r in results.values()],
                                                   hovertemplate='eps=%{x}<br>min_samples=%{text}<br>Silhouette=%{y}<extra></extra>'))
                            fig.update_layout(title='Зависимость Silhouette Score от eps',
                                            xaxis_title='eps', yaxis_title='Silhouette Score')
                            st.plotly_chart(fig, use_container_width=True)
                
                elif method_name == 'hierarchical':
                    if st.button("Подобрать оптимальные параметры"):
                        with st.spinner("Подбор параметров..."):
                            n_clusters_range = list(range(2, min(11, len(st.session_state.articles) // 10 + 1)))
                            linkage_options = ['ward', 'complete', 'average']
                            results = tune_hierarchical(st.session_state.vectors,
                                                       n_clusters_range=n_clusters_range,
                                                       linkage_options=linkage_options)
                            
                            # Создаем DataFrame
                            df_tune = pd.DataFrame(results).T
                            df_tune = df_tune.reset_index()
                            df_tune = df_tune.sort_values('silhouette', ascending=False)
                            
                            st.dataframe(df_tune, use_container_width=True)
                            
                            # График зависимости от n_clusters
                            n_clusters_values = [r['n_clusters'] for r in results.values()]
                            silhouette_values = [r['silhouette'] for r in results.values()]
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=n_clusters_values, y=silhouette_values,
                                                   mode='markers', name='Silhouette',
                                                   text=[f"linkage={r['linkage']}" for r in results.values()],
                                                   hovertemplate='n_clusters=%{x}<br>linkage=%{text}<br>Silhouette=%{y}<extra></extra>'))
                            fig.update_layout(title='Зависимость Silhouette Score от n_clusters',
                                            xaxis_title='n_clusters', yaxis_title='Silhouette Score')
                            st.plotly_chart(fig, use_container_width=True)
                
                elif method_name == 'gmm':
                    if st.button("Подобрать оптимальное количество компонент"):
                        with st.spinner("Подбор параметров..."):
                            n_components_range = list(range(2, min(11, len(st.session_state.articles) // 10 + 1)))
                            results = tune_gmm(st.session_state.vectors, n_components_range=n_components_range)
                            
                            # Создаем DataFrame
                            df_tune = pd.DataFrame(results).T
                            df_tune.index.name = 'n_components'
                            df_tune = df_tune.reset_index()
                            
                            st.dataframe(df_tune, use_container_width=True)
                            
                            # График
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_tune['n_components'], y=df_tune['silhouette'],
                                                   mode='lines+markers', name='Silhouette'))
                            fig.update_layout(title='Зависимость Silhouette Score от n_components',
                                            xaxis_title='n_components', yaxis_title='Silhouette Score')
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # График AIC и BIC
                            fig2 = go.Figure()
                            fig2.add_trace(go.Scatter(x=df_tune['n_components'], y=df_tune['aic'],
                                                    mode='lines+markers', name='AIC'))
                            fig2.add_trace(go.Scatter(x=df_tune['n_components'], y=df_tune['bic'],
                                                    mode='lines+markers', name='BIC'))
                            fig2.update_layout(title='Зависимость AIC и BIC от n_components',
                                             xaxis_title='n_components', yaxis_title='Значение')
                            st.plotly_chart(fig2, use_container_width=True)
                
                elif method_name == 'spectral':
                    if st.button("Подобрать оптимальное k"):
                        with st.spinner("Подбор параметров..."):
                            n_clusters_range = list(range(2, min(11, len(st.session_state.articles) // 10 + 1)))
                            results = tune_spectral(st.session_state.vectors, n_clusters_range=n_clusters_range)
                            
                            # Создаем DataFrame
                            df_tune = pd.DataFrame(results).T
                            df_tune.index.name = 'n_clusters'
                            df_tune = df_tune.reset_index()
                            
                            st.dataframe(df_tune, use_container_width=True)
                            
                            # График
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=df_tune['n_clusters'], y=df_tune['silhouette'],
                                                   mode='lines+markers', name='Silhouette'))
                            fig.update_layout(title='Зависимость Silhouette Score от n_clusters',
                                            xaxis_title='n_clusters', yaxis_title='Silhouette Score')
                            st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Выполните кластеризацию для подбора параметров")
        else:
            st.info("Выполните кластеризацию для просмотра метрик")
    
    with tab3:
        st.header("Интерпретация кластеров")
        
        if st.session_state.labels is not None and st.session_state.tokenized_texts:
            interpreter = ClusteringInterpreter(st.session_state.vectorizer)
            interpreter.set_data(
                st.session_state.tokenized_texts,
                st.session_state.vectors,
                st.session_state.labels
            )
            
            n_words = st.slider("Количество топ слов", 5, 20, 10)
            
            if st.button("Получить топ слова"):
                with st.spinner("Анализ кластеров..."):
                    # Получаем топ слова по частоте
                    top_words = interpreter.get_top_words_per_cluster_frequency(n_words=n_words)
                    
                    # Вычисляем размеры кластеров
                    unique_labels, counts = np.unique(st.session_state.labels, return_counts=True)
                    cluster_sizes = dict(zip(unique_labels, counts))
                    
                    # Выводим результаты
                    for cluster_id in sorted(top_words.keys()):
                        with st.expander(f"Кластер {cluster_id} (размер: {cluster_sizes.get(cluster_id, 0)} документов)"):
                            words_df = pd.DataFrame(
                                top_words[cluster_id],
                                columns=['Слово', 'Частота']
                            )
                            st.dataframe(words_df, use_container_width=True)
        else:
            st.info("Выполните кластеризацию для интерпретации")
    
    with tab4:
        st.header("Просмотр данных")
        
        if st.session_state.articles:
            article_idx = st.slider("Выберите статью", 0, len(st.session_state.articles) - 1, 0)
            
            article = st.session_state.articles[article_idx]
            
            st.subheader("Исходная статья")
            st.json(article)
            
            if st.session_state.tokenized_texts:
                st.subheader("Предобработанный текст")
                st.write(" ".join(st.session_state.tokenized_texts[article_idx]))
            
            if st.session_state.labels is not None:
                st.subheader("Кластер")
                st.write(f"**Кластер {st.session_state.labels[article_idx]}**")
else:
    st.info("👈 Загрузите JSONL файл в боковой панели для начала работы")

