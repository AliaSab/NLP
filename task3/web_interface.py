"""
Веб-интерфейс для анализа классификаторов.
Этап 8: Разработка веб-интерфейса.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import joblib
import logging
import os
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# Импорты наших модулей
from text_preprocessing import TextPreprocessor, Vectorizer, FeatureExtractor, prepare_features
from classical_classifiers import ClassicalClassifier
from neural_classifiers import NeuralClassifier
from model_interpretation import (
    FeatureImportanceAnalyzer, SHAPExplainer, LIMEExplainer,
    VisualizationTools, ErrorAnalysis
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройка страницы
st.set_page_config(
    page_title="Анализ классификаторов текстов",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Анализ классификаторов текстов")
st.markdown("---")

# Боковая панель для навигации
st.sidebar.title("Навигация")
page = st.sidebar.selectbox(
    "Выберите раздел",
    ["Интерактивная классификация", "Анализ и интерпретация", "Сравнение моделей"]
)

# Поиск доступных моделей
def find_available_models():
    """Поиск доступных обученных моделей."""
    models_info = {}
    results_dir = "results"
    
    if not os.path.exists(results_dir):
        return models_info
    
    for root, dirs, files in os.walk(results_dir):
        # Ищем папки с моделями
        if 'models' in dirs:
            task_type = os.path.basename(root)
            models_path = os.path.join(root, 'models')
            metadata_path = os.path.join(root, 'metadata.json')
            
            # Загружаем метаданные
            metadata = {}
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                except:
                    pass
            
            # Ищем модели
            available_models = []
            if os.path.exists(models_path):
                for model_file in os.listdir(models_path):
                    if model_file.endswith('.pkl'):
                        model_name = model_file.replace('.pkl', '')
                        model_path = os.path.join(models_path, model_file)
                        available_models.append({
                            'name': model_name,
                            'path': model_path
                        })
            
            if available_models:
                models_info[task_type] = {
                    'models': available_models,
                    'metadata': metadata,
                    'vectorizer_path': os.path.join(root, 'vectorizer.pkl'),
                    'label_encoder_path': os.path.join(root, 'label_encoder.pkl'),
                    'results_path': os.path.join(root, 'classical_models_results.json')
                }
    
    return models_info

# Загрузка моделей (кэширование)
@st.cache_resource
def load_model(model_path: str):
    """Загрузка одной модели."""
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        logger.error(f"Ошибка загрузки модели {model_path}: {e}")
        return None

@st.cache_resource
def load_vectorizer(vectorizer_path: str):
    """Загрузка векторизатора."""
    try:
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
            return vectorizer
        return None
    except Exception as e:
        logger.error(f"Ошибка загрузки векторизатора: {e}")
        return None

@st.cache_resource
def load_label_encoder(label_encoder_path: str):
    """Загрузка label encoder."""
    try:
        if os.path.exists(label_encoder_path):
            label_encoder = joblib.load(label_encoder_path)
            return label_encoder
        return None
    except Exception as e:
        logger.error(f"Ошибка загрузки label encoder: {e}")
        return None

@st.cache_data
def load_data(file_path: str):
    """Загрузка данных."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except Exception as e:
        logger.error(f"Ошибка загрузки данных: {e}")
    return data

# Инициализация компонентов
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = TextPreprocessor()
if 'feature_extractor' not in st.session_state:
    st.session_state.feature_extractor = FeatureExtractor()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = VisualizationTools()

# Поиск доступных моделей
available_models_info = find_available_models()

# Главная страница - Интерактивная классификация
if page == "Интерактивная классификация":
    st.header("🔍 Интерактивная классификация")
    
    if not available_models_info:
        st.warning("⚠️ Обученные модели не найдены. Пожалуйста, сначала запустите обучение через run_pipeline.py")
        st.info("После обучения модели будут доступны в разделе 'Интерактивная классификация'")
        
        st.markdown("### 📝 Инструкция по обучению моделей:")
        st.code("""
# Бинарная классификация (уже выполнено):
python run_pipeline.py --train corpus/binary_classification/train.jsonl --val corpus/binary_classification/validation.jsonl --test corpus/binary_classification/test.jsonl --task binary --output results/binary

# Многоклассовая классификация:
python run_pipeline.py --train corpus/multiclass_classification/train.jsonl --val corpus/multiclass_classification/validation.jsonl --test corpus/multiclass_classification/test.jsonl --task multiclass --output results/multiclass

# Многометочная классификация:
python run_pipeline.py --train corpus/multilabel_classification/train.jsonl --val corpus/multilabel_classification/validation.jsonl --test corpus/multilabel_classification/test.jsonl --task multilabel --output results/multilabel
        """, language="bash")
    else:
        # Выбор задачи и модели
        task_types = list(available_models_info.keys())
        selected_task = st.selectbox(
            "Выберите задачу",
            task_types
        )
        
        task_info = available_models_info[selected_task]
        available_models = task_info['models']
        
        if not available_models:
            st.warning("Модели не найдены для выбранной задачи")
        else:
            model_names = [m['name'] for m in available_models]
            selected_model_name = st.selectbox(
                "Выберите модель",
                model_names
            )
            
            # Загрузка модели и компонентов
            selected_model_info = next(m for m in available_models if m['name'] == selected_model_name)
            model = load_model(selected_model_info['path'])
            vectorizer = load_vectorizer(task_info['vectorizer_path'])
            label_encoder = load_label_encoder(task_info['label_encoder_path'])
            
            if model is None:
                st.error("Ошибка загрузки модели")
            elif vectorizer is None:
                st.error("Ошибка загрузки векторизатора")
            else:
                text_input = st.text_area(
                    "Введите текст для классификации",
                    height=200,
                    placeholder="Введите текст здесь..."
                )

                if st.button("Классифицировать", type="primary"):
                    if text_input:
                        with st.spinner("Обработка текста..."):
                            try:
                                tokens = st.session_state.preprocessor.preprocess(text_input)

                                # Векторизация
                                full_text = text_input
                                texts = [full_text]

                                # Подготовка признаков
                                feature_vectors = []

                                # TF-IDF
                                if vectorizer.tfidf_vectorizer is not None:
                                    try:
                                        tfidf_features = vectorizer.transform_tfidf(texts)
                                        feature_vectors.append(tfidf_features)
                                    except Exception as e:
                                        st.warning(f"Ошибка TF-IDF векторизации: {e}")

                                # Статистические признаки
                                stat_features = []
                                stat_feat = st.session_state.feature_extractor.extract_statistical_features(full_text)
                                stat_features.append(list(stat_feat.values()))
                                feature_vectors.append(np.array(stat_features))

                                # Объединение признаков
                                if len(feature_vectors) > 1:
                                    X = np.hstack(feature_vectors)
                                else:
                                    X = feature_vectors[0]

                                # Предсказание
                                if hasattr(model, 'predict_proba'):
                                    y_proba = model.predict_proba(X)[0]
                                    y_pred = model.predict(X)[0]
                                else:
                                    y_pred = model.predict(X)[0]
                                    # Если нет predict_proba, создаем фиктивные вероятности
                                    if label_encoder:
                                        num_classes = len(label_encoder.classes_)
                                        y_proba = np.zeros(num_classes)
                                        y_proba[y_pred] = 1.0
                                    else:
                                        y_proba = np.array([0.5, 0.5])

                                st.success("Классификация выполнена!")

                                # Результаты
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.subheader("Результаты")

                                    # Определяем имена классов
                                    if label_encoder:
                                        class_names = label_encoder.classes_
                                        predicted_class = class_names[y_pred]

                                        if len(class_names) == 2:  # Бинарная
                                            st.metric("Тональность", predicted_class, delta=f"{y_proba[y_pred]:.2f}")
                                        else:  # Многоклассовая
                                            st.metric("Категория", predicted_class, delta=f"{y_proba[y_pred]:.2f}")
                                    else:
                                        st.metric("Класс", f"Класс {y_pred}", delta=f"{y_proba[y_pred]:.2f}")

                                with col2:
                                    st.subheader("Вероятности")
                                    # График вероятностей
                                    if label_encoder:
                                        categories = [str(c) for c in label_encoder.classes_]
                                    else:
                                        categories = [f"Класс {i}" for i in range(len(y_proba))]

                                    fig, ax = plt.subplots(figsize=(8, 5))
                                    bars = ax.barh(categories, y_proba)
                                    ax.set_xlabel('Вероятность')
                                    ax.set_title('Распределение вероятностей')
                                    ax.set_xlim([0, 1])

                                    # Подсветка предсказанного класса
                                    if label_encoder:
                                        pred_idx = y_pred
                                        bars[pred_idx].set_color('green')

                                    plt.tight_layout()
                                    st.pyplot(fig)

                            except Exception as e:
                                st.error(f"Ошибка классификации: {e}")
                                logger.exception(e)
                    else:
                        st.warning("Пожалуйста, введите текст для классификации")

# Страница анализа и интерпретации
elif page == "Анализ и интерпретация":
    st.header("🔬 Анализ и интерпретация")

    if not available_models_info:
        st.warning("⚠️ Обученные модели не найдены. Пожалуйста, сначала запустите обучение")

        st.markdown("### 📝 Инструкция по обучению моделей:")
        st.code("""
# Бинарная классификация:
python run_pipeline.py --train corpus/binary_classification/train.jsonl --val corpus/binary_classification/validation.jsonl --test corpus/binary_classification/test.jsonl --task binary --output results/binary

# Многоклассовая классификация:
python run_pipeline.py --train corpus/multiclass_classification/train.jsonl --val corpus/multiclass_classification/validation.jsonl --test corpus/multiclass_classification/test.jsonl --task multiclass --output results/multiclass

# Многометочная классификация:
python run_pipeline.py --train corpus/multilabel_classification/train.jsonl --val corpus/multilabel_classification/validation.jsonl --test corpus/multilabel_classification/test.jsonl --task multilabel --output results/multilabel
        """, language="bash")
    else:
        tab1, tab2 = st.tabs(["SHAP/LIME объяснения", "Анализ ошибок"])

        with tab1:
            st.subheader("SHAP/LIME объяснения")
            task_types = list(available_models_info.keys())
            selected_task = st.selectbox(
                "Выберите задачу",
                task_types,
                key="explain_task"
            )

            task_info = available_models_info[selected_task]
            available_models = task_info['models']

            if available_models:
                model_names = [m['name'] for m in available_models]
                selected_model_name = st.selectbox(
                    "Выберите модель",
                    model_names,
                    key="explain_model"
                )
                selected_model_info = next(m for m in available_models if m['name'] == selected_model_name)
                model = load_model(selected_model_info['path'])
                vectorizer = load_vectorizer(task_info['vectorizer_path'])
                label_encoder = load_label_encoder(task_info['label_encoder_path'])

            explain_text = st.text_area(
                "Введите текст для объяснения",
                height=150,
                key="explain_text"
            )

            explain_method = st.selectbox(
                "Метод объяснения",
                ["SHAP", "LIME"],
                key="explain_method"
            )

            if st.button("Объяснить", key="explain_button"):
                if explain_text and model and vectorizer:
                    with st.spinner("Вычисление объяснения..."):
                        try:
                            # Подготовка текста
                            texts = [explain_text]

                            # Векторизация
                            feature_vectors = []
                            if vectorizer.tfidf_vectorizer is not None:
                                try:
                                    tfidf_features = vectorizer.transform_tfidf(texts)
                                    # Преобразуем sparse matrix в dense, если нужно
                                    if hasattr(tfidf_features, 'toarray'):
                                        tfidf_features = tfidf_features.toarray()
                                    feature_vectors.append(tfidf_features)
                                except Exception as e:
                                    logger.warning(f"Ошибка TF-IDF: {e}")

                            stat_features = []
                            stat_feat = st.session_state.feature_extractor.extract_statistical_features(explain_text)
                            stat_features.append(list(stat_feat.values()))
                            feature_vectors.append(np.array(stat_features))

                            if len(feature_vectors) > 1:
                                X = np.hstack(feature_vectors)
                            else:
                                X = feature_vectors[0]

                            # Убеждаемся, что X - это numpy array
                            if hasattr(X, 'toarray'):
                                X = X.toarray()
                            X = np.array(X)

                            # Для SHAP нужен background dataset - создаем из нескольких примеров
                            if explain_method == "SHAP":
                                try:
                                    # Проверяем, поддерживает ли модель SHAP
                                    from sklearn.ensemble import VotingClassifier
                                    if isinstance(model, VotingClassifier):
                                        st.warning("⚠️ SHAP объяснения не поддерживаются для ансамблевых моделей (stacking/voting). Используйте LIME или выберите другую модель.")
                                    else:
                                        # SHAP объяснение
                                        # Создаем background dataset из нескольких примеров
                                        # Используем более разнообразный background dataset
                                        background_size = min(50, max(10, X.shape[0] if len(X.shape) > 0 else 10))
                                        
                                        # Создаем background из вариаций исходного текста
                                        background_vectors = []
                                        
                                        # Для background используем тот же вектор, но с небольшими вариациями
                                        if vectorizer.tfidf_vectorizer is not None:
                                            # Создаем background из нулевых векторов и исходного вектора
                                            bg_tfidf_base = vectorizer.transform_tfidf([explain_text])
                                            if hasattr(bg_tfidf_base, 'toarray'):
                                                bg_tfidf_base = bg_tfidf_base.toarray()
                                            
                                            # Создаем background dataset
                                            bg_tfidf_list = []
                                            for i in range(background_size):
                                                if i == 0:
                                                    bg_tfidf_list.append(bg_tfidf_base[0])
                                                else:
                                                    # Добавляем небольшие вариации или нулевой вектор
                                                    bg_tfidf_list.append(np.zeros_like(bg_tfidf_base[0]))
                                            bg_tfidf = np.array(bg_tfidf_list)
                                            background_vectors.append(bg_tfidf)

                                        bg_stat = []
                                        stat_feat_base = st.session_state.feature_extractor.extract_statistical_features(explain_text)
                                        for i in range(background_size):
                                            if i == 0:
                                                bg_stat.append(list(stat_feat_base.values()))
                                            else:
                                                # Используем те же статистические признаки или нулевые
                                                bg_stat.append(list(stat_feat_base.values()))
                                        background_vectors.append(np.array(bg_stat))

                                        if len(background_vectors) > 1:
                                            X_background = np.hstack(background_vectors)
                                        else:
                                            X_background = background_vectors[0]

                                        if hasattr(X_background, 'toarray'):
                                            X_background = X_background.toarray()
                                        X_background = np.array(X_background)

                                        # Используем только первый пример для объяснения
                                        X_explain = X[0:1] if len(X.shape) > 1 else X.reshape(1, -1)

                                        # Проверяем доступность SHAP
                                        from model_interpretation import SHAP_AVAILABLE
                                        if not SHAP_AVAILABLE:
                                            st.error("❌ Библиотека SHAP не установлена. Установите её командой: pip install shap")
                                        else:
                                            shap_explainer = SHAPExplainer(model, X_background, vectorizer.tfidf_vectorizer)
                                            shap_values = shap_explainer.explain(X_explain)

                                            if shap_values is not None:
                                                st.success("✅ SHAP объяснение вычислено")

                                                # Получаем важные признаки
                                                if vectorizer.tfidf_vectorizer is not None:
                                                    try:
                                                        feature_names = vectorizer.tfidf_vectorizer.get_feature_names_out()

                                                        # Для SHAP получаем значения важности
                                                        # SHAP значения могут быть в разных форматах
                                                        importance_scores = None
                                                        
                                                        # Проверяем, является ли shap_values списком (для многоклассовой классификации)
                                                        if isinstance(shap_values, list):
                                                            # Для многоклассовой классификации берем среднее по всем классам
                                                            shap_array = np.array([np.array(sv) for sv in shap_values])
                                                            # Берем абсолютные значения и усредняем по классам
                                                            importance_scores = np.abs(shap_array).mean(axis=0)
                                                            # Если это массив с дополнительными измерениями, берем первый пример
                                                            if len(importance_scores.shape) > 1:
                                                                importance_scores = importance_scores[0] if importance_scores.shape[0] == 1 else importance_scores.mean(axis=0)
                                                        elif hasattr(shap_values, 'values'):
                                                            # SHAP Explanation объект
                                                            sv_values = shap_values.values
                                                            # Обрабатываем многомерные массивы
                                                            if isinstance(sv_values, np.ndarray):
                                                                if len(sv_values.shape) > 2:
                                                                    # Многоклассовая: [n_samples, n_features, n_classes]
                                                                    importance_scores = np.abs(sv_values[0]).mean(axis=-1)
                                                                elif len(sv_values.shape) == 2:
                                                                    # [n_samples, n_features] или [n_features, n_classes]
                                                                    if sv_values.shape[0] == 1:
                                                                        importance_scores = np.abs(sv_values[0])
                                                                    else:
                                                                        # Возможно, это [n_features, n_classes]
                                                                        importance_scores = np.abs(sv_values).mean(axis=-1) if sv_values.shape[1] > 1 else np.abs(sv_values[0])
                                                                else:
                                                                    importance_scores = np.abs(sv_values)
                                                            else:
                                                                importance_scores = np.abs(np.array(sv_values).flatten())
                                                        elif hasattr(shap_values, 'data'):
                                                            # SHAP Explanation объект с data
                                                            sv_data = shap_values.data
                                                            if isinstance(sv_data, np.ndarray) and len(sv_data.shape) > 1:
                                                                importance_scores = np.abs(sv_data[0])
                                                            else:
                                                                importance_scores = np.abs(np.array(sv_data).flatten())
                                                        else:
                                                            # Пытаемся преобразовать в numpy array
                                                            sv_array = np.array(shap_values)
                                                            if len(sv_array.shape) > 2:
                                                                # Многоклассовая: усредняем по классам
                                                                importance_scores = np.abs(sv_array).mean(axis=-1)
                                                                if len(importance_scores.shape) > 1:
                                                                    importance_scores = importance_scores[0]
                                                            elif len(sv_array.shape) == 2:
                                                                # [n_samples, n_features] или [n_features, n_classes]
                                                                if sv_array.shape[0] == 1:
                                                                    importance_scores = np.abs(sv_array[0])
                                                                else:
                                                                    # Возможно, это [n_features, n_classes]
                                                                    importance_scores = np.abs(sv_array).mean(axis=-1) if sv_array.shape[1] > 1 else np.abs(sv_array[0])
                                                            else:
                                                                importance_scores = np.abs(sv_array)

                                                        # Убеждаемся, что importance_scores - это одномерный массив
                                                        if importance_scores is not None:
                                                            importance_scores = np.array(importance_scores).flatten()
                                                            
                                                            # Берем топ признаков
                                                            if len(importance_scores) > 0:
                                                                top_indices = np.argsort(importance_scores)[-20:][::-1]

                                                                st.write("**Важные слова/признаки:**")
                                                                important_features = []

                                                                # Разделяем TF-IDF признаки и статистические
                                                                tfidf_dim = len(feature_names) if vectorizer.tfidf_vectorizer is not None else 0

                                                                for idx in top_indices:
                                                                    if idx < len(importance_scores) and idx < tfidf_dim:
                                                                        # Это TF-IDF признак
                                                                        feat_name = feature_names[idx]
                                                                        importance = float(importance_scores[idx])
                                                                        important_features.append((feat_name, importance))
                                                                    # Статистические признаки пропускаем для краткости

                                                                # Сортируем по важности
                                                                important_features.sort(key=lambda x: x[1], reverse=True)

                                                                for feat_name, importance in important_features[:15]:
                                                                    st.write(f"- {feat_name} (важность: {importance:.4f})")
                                                            else:
                                                                st.warning("Не удалось извлечь важность признаков")
                                                        else:
                                                            st.warning("Не удалось обработать SHAP значения")
                                                    except Exception as e:
                                                        st.warning(f"Ошибка при обработке SHAP значений: {e}")
                                                        logger.exception(e)
                                            else:
                                                st.error("❌ Не удалось вычислить SHAP значения. Попробуйте использовать LIME или выберите другую модель.")
                                except ImportError as e:
                                    st.error(f"❌ Ошибка импорта: {e}. Убедитесь, что библиотека SHAP установлена: pip install shap")
                                except Exception as e:
                                    error_msg = str(e)
                                    if "VotingClassifier" in error_msg or "ensemble" in error_msg.lower():
                                        st.warning("⚠️ SHAP объяснения не поддерживаются для ансамблевых моделей (stacking/voting). Используйте LIME или выберите другую модель.")
                                    else:
                                        st.error(f"❌ Ошибка SHAP объяснения: {error_msg}")
                                    logger.exception(e)

                            elif explain_method == "LIME":
                                try:
                                    # Проверяем доступность LIME
                                    from model_interpretation import LIME_AVAILABLE
                                    if not LIME_AVAILABLE:
                                        st.error("❌ Библиотека LIME не установлена. Установите её командой: pip install lime")
                                    else:
                                        # LIME объяснение
                                        label_encoder = load_label_encoder(task_info['label_encoder_path'])
                                        class_names = None
                                        if label_encoder:
                                            class_names = [str(c) for c in label_encoder.classes_]
                                        else:
                                            # Определяем количество классов из модели
                                            if hasattr(model, 'classes_'):
                                                class_names = [str(c) for c in model.classes_]
                                            else:
                                                # Пробуем предсказать, чтобы определить количество классов
                                                test_pred = model.predict(X[0:1] if len(X.shape) > 1 else X.reshape(1, -1))
                                                if isinstance(test_pred, np.ndarray):
                                                    num_classes = len(np.unique(test_pred)) if len(test_pred.shape) == 1 else test_pred.shape[1]
                                                else:
                                                    num_classes = 2
                                                class_names = [f"Класс {i}" for i in range(num_classes)]

                                        lime_explainer = LIMEExplainer(model, [explain_text], class_names)

                                        # Создаем функцию предсказания для LIME
                                        def predict_proba_lime(texts_list):
                                            """Функция предсказания для LIME, которая работает с текстами."""
                                            try:
                                                # Векторизуем тексты
                                                all_features = []

                                                for t in texts_list:
                                                    if not isinstance(t, str):
                                                        t = str(t)
                                                    
                                                    feature_vectors_list = []

                                                    # TF-IDF
                                                    if vectorizer.tfidf_vectorizer is not None:
                                                        try:
                                                            tfidf = vectorizer.transform_tfidf([t])
                                                            if hasattr(tfidf, 'toarray'):
                                                                tfidf = tfidf.toarray()
                                                            # Убеждаемся, что это 2D массив
                                                            if len(tfidf.shape) == 1:
                                                                tfidf = tfidf.reshape(1, -1)
                                                            feature_vectors_list.append(tfidf)
                                                        except Exception as e:
                                                            logger.warning(f"Ошибка TF-IDF для LIME: {e}")
                                                            # Создаем нулевой вектор той же размерности
                                                            if vectorizer.tfidf_vectorizer is not None:
                                                                try:
                                                                    vocab_size = len(vectorizer.tfidf_vectorizer.vocabulary_)
                                                                except:
                                                                    vocab_size = len(vectorizer.tfidf_vectorizer.get_feature_names_out())
                                                                feature_vectors_list.append(np.zeros((1, vocab_size)))
                                                            else:
                                                                feature_vectors_list.append(np.zeros((1, 1000)))

                                                    # Статистические признаки
                                                    try:
                                                        stat_feat = st.session_state.feature_extractor.extract_statistical_features(t)
                                                        stat_feat_array = np.array([list(stat_feat.values())])
                                                        # Убеждаемся, что это 2D массив
                                                        if len(stat_feat_array.shape) == 1:
                                                            stat_feat_array = stat_feat_array.reshape(1, -1)
                                                        feature_vectors_list.append(stat_feat_array)
                                                    except Exception as e:
                                                        logger.warning(f"Ошибка статистических признаков для LIME: {e}")
                                                        # Создаем нулевой вектор для статистических признаков
                                                        feature_vectors_list.append(np.zeros((1, 10)))

                                                    # Объединяем признаки
                                                    if len(feature_vectors_list) > 1:
                                                        try:
                                                            X_single = np.hstack(feature_vectors_list)
                                                        except Exception as e:
                                                            logger.warning(f"Ошибка объединения признаков: {e}")
                                                            # Пробуем выровнять размерности
                                                            max_len = max(fv.shape[1] if len(fv.shape) > 1 else len(fv) for fv in feature_vectors_list)
                                                            aligned_features = []
                                                            for fv in feature_vectors_list:
                                                                if len(fv.shape) == 1:
                                                                    fv = fv.reshape(1, -1)
                                                                if fv.shape[1] < max_len:
                                                                    fv = np.pad(fv, ((0, 0), (0, max_len - fv.shape[1])), mode='constant')
                                                                elif fv.shape[1] > max_len:
                                                                    fv = fv[:, :max_len]
                                                                aligned_features.append(fv)
                                                            X_single = np.hstack(aligned_features)
                                                    elif len(feature_vectors_list) == 1:
                                                        X_single = feature_vectors_list[0]
                                                        if len(X_single.shape) == 1:
                                                            X_single = X_single.reshape(1, -1)
                                                    else:
                                                        # Fallback - создаем фиктивный вектор
                                                        X_single = np.zeros((1, 100))

                                                    all_features.append(X_single)

                                                # Объединяем все в один массив
                                                if all_features:
                                                    try:
                                                        X_list = np.vstack(all_features)
                                                    except Exception as e:
                                                        logger.warning(f"Ошибка vstack: {e}")
                                                        # Пробуем выровнять размерности
                                                        max_len = max(f.shape[1] if len(f.shape) > 1 else len(f) for f in all_features)
                                                        aligned_features = []
                                                        for f in all_features:
                                                            if len(f.shape) == 1:
                                                                f = f.reshape(1, -1)
                                                            if f.shape[1] < max_len:
                                                                f = np.pad(f, ((0, 0), (0, max_len - f.shape[1])), mode='constant')
                                                            elif f.shape[1] > max_len:
                                                                f = f[:, :max_len]
                                                            aligned_features.append(f)
                                                        X_list = np.vstack(aligned_features)
                                                else:
                                                    # Fallback
                                                    num_classes = len(class_names) if class_names else 2
                                                    return np.ones((len(texts_list), num_classes)) / num_classes

                                                # Убеждаемся, что X_list правильной формы
                                                if len(X_list.shape) == 1:
                                                    X_list = X_list.reshape(1, -1)

                                                # Предсказание
                                                if hasattr(model, 'predict_proba'):
                                                    try:
                                                        probs = model.predict_proba(X_list)
                                                        # Убеждаемся, что вероятности нормализованы
                                                        if len(probs.shape) == 1:
                                                            probs = probs.reshape(1, -1)
                                                        if probs.shape[1] > 1:
                                                            row_sums = probs.sum(axis=1, keepdims=True)
                                                            row_sums[row_sums == 0] = 1  # Избегаем деления на ноль
                                                            probs = probs / row_sums
                                                        return probs
                                                    except Exception as e:
                                                        logger.warning(f"Ошибка predict_proba: {e}, используем predict")
                                                        # Если ошибка, используем predict
                                                        preds = model.predict(X_list)
                                                        num_classes = len(class_names) if class_names else 2
                                                        probs = np.zeros((len(preds), num_classes))
                                                        for i, p in enumerate(preds):
                                                            p_idx = int(p) if isinstance(p, (np.integer, np.int64)) else p
                                                            if 0 <= p_idx < num_classes:
                                                                probs[i, p_idx] = 1.0
                                                        return probs
                                                else:
                                                    # Фиктивные вероятности на основе predict
                                                    preds = model.predict(X_list)
                                                    num_classes = len(class_names) if class_names else 2
                                                    probs = np.zeros((len(preds), num_classes))
                                                    for i, p in enumerate(preds):
                                                        p_idx = int(p) if isinstance(p, (np.integer, np.int64)) else p
                                                        if 0 <= p_idx < num_classes:
                                                            probs[i, p_idx] = 1.0
                                                    return probs
                                            except Exception as e:
                                                logger.error(f"Ошибка в predict_proba_lime: {e}")
                                                logger.exception(e)
                                                # Возвращаем равномерное распределение в случае ошибки
                                                num_classes = len(class_names) if class_names else 2
                                                return np.ones((len(texts_list), num_classes)) / num_classes

                                        explanation = lime_explainer.explain_instance(explain_text, num_features=15, predict_proba_func=predict_proba_lime)

                                        if explanation:
                                            st.success("✅ LIME объяснение вычислено")
                                            st.write("**Важные слова:**")

                                            # Получаем объяснение
                                            try:
                                                exp_list = explanation.as_list()
                                                if exp_list:
                                                    for word, importance in exp_list:
                                                        st.write(f"- {word} (важность: {importance:.4f})")
                                                else:
                                                    st.warning("LIME не вернул объяснение")
                                            except Exception as e:
                                                st.warning(f"Ошибка при получении LIME объяснения: {e}")
                                                logger.exception(e)
                                        else:
                                            st.warning("⚠️ Не удалось вычислить LIME объяснение")
                                except ImportError as e:
                                    st.error(f"❌ Ошибка импорта LIME: {e}. Убедитесь, что библиотека установлена: pip install lime")
                                except Exception as e:
                                    st.error(f"❌ Ошибка LIME объяснения: {e}")
                                    logger.exception(e)

                        except Exception as e:
                            st.error(f"Ошибка: {e}")
                            logger.exception(e)
                else:
                    st.warning("Введите текст для объяснения")
            else:
                st.warning("Модели не найдены для выбранной задачи")

        with tab2:
            st.subheader("Анализ ошибок")

            # Выбор задачи
            task_types = list(available_models_info.keys())
            selected_task = st.selectbox(
                "Выберите задачу",
                task_types,
                key="error_task"
            )

            task_info = available_models_info[selected_task]

            # Выбор модели для анализа
            available_models = task_info['models']
            if available_models:
                model_names = [m['name'] for m in available_models]
                selected_model_name = st.selectbox(
                    "Выберите модель для анализа",
                    model_names,
                    key="error_model"
                )

                # Загрузка confusion matrix
                cm_path = os.path.join(os.path.dirname(task_info['vectorizer_path']), f'{selected_model_name}_confusion_matrix.png')
                if os.path.exists(cm_path):
                    st.subheader("📊 Матрица ошибок (Confusion Matrix)")
                    st.image(cm_path, use_container_width=True)

                # Загрузка данных об ошибках
                error_analysis_path = os.path.join(os.path.dirname(task_info['vectorizer_path']), f'{selected_model_name}_error_analysis.json')

                if os.path.exists(error_analysis_path):
                    try:
                        with open(error_analysis_path, 'r', encoding='utf-8') as f:
                            error_data = json.load(f)

                        st.subheader("📈 Статистика ошибок")
                        total_errors = error_data.get('total_errors', 0)
                        error_types = error_data.get('error_types', {})

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Всего ошибок", total_errors)
                        with col2:
                            st.metric("Типов ошибок", len(error_types))

                        if error_types:
                            st.write("**Распределение по типам ошибок:**")
                            for error_type, count in error_types.items():
                                st.write(f"- {error_type}: {count}")

                        # Примеры ошибок
                        examples = error_data.get('examples', {})
                        if examples:
                            st.subheader("🔍 Примеры ошибок классификации")
                            error_count = 0
                            for error_type, error_list in examples.items():
                                for error in error_list[:5]:  # Показываем до 5 примеров каждого типа
                                    error_count += 1
                                    with st.expander(f"Ошибка {error_count}: {error_type}"):
                                        text = error.get('text', 'Текст не найден')
                                        # Ограничиваем длину текста
                                        if len(text) > 500:
                                            text = text[:500] + "..."
                                        st.write(f"**Текст:** {text}")
                                        st.write(f"**Истинная метка:** {error.get('true_label', 'N/A')}")
                                        st.write(f"**Предсказанная метка:** {error.get('predicted_label', 'N/A')}")
                        else:
                            if total_errors == 0:
                                st.success("✅ Ошибок не обнаружено! Модель работает идеально на тестовой выборке.")
                                st.warning("⚠️ **Внимание:** Если тестовая выборка очень мала (менее 10 образцов), результаты могут быть нерепрезентативными.")
                    except Exception as e:
                        st.error(f"Ошибка загрузки анализа ошибок: {e}")
                        logger.exception(e)
                else:
                    # Пытаемся сгенерировать анализ ошибок на лету
                    st.info("ℹ️ Анализ ошибок не найден для выбранной модели.")
                    
                    # Загружаем модель и компоненты
                    selected_model_info = next(m for m in available_models if m['name'] == selected_model_name)
                    model = load_model(selected_model_info['path'])
                    vectorizer = load_vectorizer(task_info['vectorizer_path'])
                    label_encoder = load_label_encoder(task_info['label_encoder_path'])
                    
                    if model and vectorizer:
                        # Пытаемся загрузить тестовые данные
                        corpus_dir = "corpus"
                        task_corpus_map = {
                            'binary': 'binary_classification',
                            'multiclass': 'multiclass_classification',
                            'multilabel': 'multilabel_classification'
                        }
                        
                        corpus_subdir = task_corpus_map.get(selected_task, selected_task)
                        test_path = os.path.join(corpus_dir, corpus_subdir, 'test.jsonl')
                        
                        if os.path.exists(test_path):
                            if st.button("🔍 Сгенерировать анализ ошибок", key="generate_error_analysis"):
                                with st.spinner("Генерация анализа ошибок..."):
                                    try:
                                        # Загружаем тестовые данные
                                        test_data = load_data(test_path)
                                        
                                        if test_data:
                                            # Подготавливаем тексты и метки
                                            texts_test = []
                                            y_test = []
                                            
                                            for item in test_data:
                                                texts_test.append(item.get('text', ''))
                                                if selected_task == 'multilabel':
                                                    # Для multilabel метки - это список
                                                    labels = item.get('categories', [])
                                                    y_test.append(labels)
                                                elif selected_task == 'multiclass':
                                                    label = item.get('category', '')
                                                    if label_encoder:
                                                        y_test.append(label)
                                                    else:
                                                        y_test.append(label)
                                                else:  # binary
                                                    label = item.get('sentiment', '')
                                                    if label_encoder:
                                                        y_test.append(label)
                                                    else:
                                                        y_test.append(label)
                                            
                                            # Векторизуем данные
                                            feature_vectors = []
                                            
                                            # TF-IDF
                                            if vectorizer.tfidf_vectorizer is not None:
                                                try:
                                                    tfidf_features = vectorizer.transform_tfidf(texts_test)
                                                    if hasattr(tfidf_features, 'toarray'):
                                                        tfidf_features = tfidf_features.toarray()
                                                    feature_vectors.append(tfidf_features)
                                                except Exception as e:
                                                    logger.warning(f"Ошибка TF-IDF: {e}")
                                            
                                            # Статистические признаки
                                            stat_features = []
                                            for text in texts_test:
                                                stat_feat = st.session_state.feature_extractor.extract_statistical_features(text)
                                                stat_features.append(list(stat_feat.values()))
                                            feature_vectors.append(np.array(stat_features))
                                            
                                            # Объединяем признаки
                                            if len(feature_vectors) > 1:
                                                X_test = np.hstack(feature_vectors)
                                            else:
                                                X_test = feature_vectors[0]
                                            
                                            if hasattr(X_test, 'toarray'):
                                                X_test = X_test.toarray()
                                            X_test = np.array(X_test)
                                            
                                            # Кодируем метки, если есть label_encoder
                                            if label_encoder:
                                                try:
                                                    y_test_encoded = label_encoder.transform(y_test)
                                                except:
                                                    # Если не получается закодировать, используем как есть
                                                    y_test_encoded = np.array(y_test)
                                            else:
                                                y_test_encoded = np.array(y_test)
                                            
                                            # Выполняем анализ ошибок
                                            from model_interpretation import ErrorAnalysis
                                            error_analyzer = ErrorAnalysis(model, X_test, y_test_encoded, texts_test)
                                            errors = error_analyzer.find_errors(n_errors=20)
                                            error_patterns = error_analyzer.analyze_error_patterns()
                                            
                                            # Сохраняем результат
                                            with open(error_analysis_path, 'w', encoding='utf-8') as f:
                                                json.dump(error_patterns, f, ensure_ascii=False, indent=2)
                                            
                                            st.success("✅ Анализ ошибок успешно сгенерирован!")
                                            st.rerun()
                                        else:
                                            st.warning("⚠️ Не удалось загрузить тестовые данные")
                                    except Exception as e:
                                        st.error(f"❌ Ошибка генерации анализа ошибок: {e}")
                                        logger.exception(e)
                        else:
                            st.info("💡 Для генерации анализа ошибок на лету нужны тестовые данные. Запустите обучение моделей через run_pipeline.py для автоматической генерации анализа ошибок.")
                    else:
                        st.warning("⚠️ Не удалось загрузить модель или векторизатор")
            else:
                st.warning("Модели не найдены для выбранной задачи")

# Страница сравнения моделей
elif page == "Сравнение моделей":
    st.header("📈 Сравнение моделей")

    # Поиск доступных JSON файлов с результатами
    results_dir = "results"
    available_results = {}

    if os.path.exists(results_dir):
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if file.endswith('_results.json') or file == 'classical_models_results.json':
                    task_type = os.path.basename(root)
                    if task_type == 'results':
                        task_type = 'general'
                    file_path = os.path.join(root, file)
                    if task_type not in available_results:
                        available_results[task_type] = []
                    available_results[task_type].append(file_path)

    # Выбор источника данных
    col1, col2 = st.columns(2)

    with col1:
        if available_results:
            st.subheader("Доступные результаты")
            selected_task = st.selectbox(
                "Выберите задачу",
                list(available_results.keys()) + ["Загрузить файл"]
            )
        else:
            selected_task = "Загрузить файл"
            st.info("Локальные результаты не найдены")

    with col2:
        results_file = st.file_uploader("Или загрузите JSON файл", type=['json'])

    # Загрузка данных
    results = None

    if selected_task != "Загрузить файл" and selected_task in available_results:
        # Загрузка из локального файла
        if available_results[selected_task]:
            selected_file = st.selectbox(
                "Выберите файл",
                available_results[selected_task],
                format_func=lambda x: os.path.basename(x)
            )

            if st.button("Загрузить результаты"):
                try:
                    with open(selected_file, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    st.success(f"Загружено из: {os.path.basename(selected_file)}")
                except Exception as e:
                    st.error(f"Ошибка загрузки: {e}")

    elif results_file:
        # Загрузка из загруженного файла
        try:
            results = json.load(results_file)
            st.success("Файл загружен успешно!")
        except Exception as e:
            st.error(f"Ошибка загрузки файла: {e}")

    # Отображение результатов
    if results:
        # Предупреждение о малой выборке
        # Проверяем размер confusion matrix для определения размера тестовой выборки
        sample_sizes = []
        for model_name, model_results in results.items():
            if isinstance(model_results, dict) and 'confusion_matrix' in model_results:
                cm = model_results['confusion_matrix']
                if isinstance(cm, list):
                    total_samples = sum(sum(row) for row in cm)
                    sample_sizes.append(total_samples)

        if sample_sizes and max(sample_sizes) < 10:
            st.warning(f"⚠️ **Внимание:** Тестовая выборка очень мала ({max(sample_sizes)} образцов). Результаты могут быть нерепрезентативными. Рекомендуется использовать выборку не менее 100-200 образцов для достоверной оценки.")

        # Таблица сравнения
        st.subheader("📊 Таблица сравнения метрик")

        # Преобразование в DataFrame
        try:
            df = pd.DataFrame(results).T

            # Очистка данных - удаляем вложенные структуры
            clean_df = pd.DataFrame()
            for col in df.columns:
                if isinstance(df[col].iloc[0], (int, float, str)):
                    clean_df[col] = df[col]
                elif isinstance(df[col].iloc[0], dict):
                    # Извлекаем основные метрики из словаря
                    for key in df[col].iloc[0].keys():
                        if isinstance(df[col].iloc[0][key], (int, float)):
                            clean_df[f"{col}_{key}"] = df[col].apply(lambda x: x.get(key, None) if isinstance(x, dict) else None)

            if not clean_df.empty:
                st.dataframe(clean_df, use_container_width=True)

                # Визуализация
                st.subheader("📈 Визуализация результатов")

                # Выбор метрик для отображения
                numeric_cols = clean_df.select_dtypes(include=[np.number]).columns.tolist()

                if numeric_cols:
                    metric = st.selectbox(
                        "Метрика для сравнения",
                        numeric_cols
                    )

                    if metric in clean_df.columns:
                        fig, ax = plt.subplots(figsize=(12, 6))
                        clean_df[metric].plot(kind='bar', ax=ax, color='steelblue')
                        ax.set_ylabel(metric)
                        ax.set_title(f'Сравнение моделей по метрике {metric}')
                        ax.tick_params(axis='x', rotation=45)
                        ax.grid(True, alpha=0.3)
                        plt.tight_layout()
                        st.pyplot(fig)

                        # Статистика
                        st.subheader("📋 Статистика")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Лучшая модель", clean_df[metric].idxmax())
                        with col2:
                            st.metric("Лучший результат", f"{clean_df[metric].max():.4f}")
                        with col3:
                            st.metric("Средний результат", f"{clean_df[metric].mean():.4f}")
                else:
                    st.warning("Не найдено числовых метрик для визуализации")
            else:
                st.warning("Не удалось преобразовать данные в таблицу")
                st.json(results)

        except Exception as e:
            st.error(f"Ошибка обработки данных: {e}")
            st.json(results)

        # ROC-кривые (если есть данные)
        if any('roc_auc' in str(results.get(k, {})).lower() for k in results.keys()):
            st.subheader("📉 ROC-AUC значения")
            roc_data = {}
            for model_name, model_results in results.items():
                if isinstance(model_results, dict) and 'roc_auc' in model_results:
                    roc_data[model_name] = model_results['roc_auc']

            if roc_data:
                fig, ax = plt.subplots(figsize=(10, 6))
                models = list(roc_data.keys())
                values = list(roc_data.values())
                ax.barh(models, values, color='green', alpha=0.7)
                ax.set_xlabel('ROC-AUC')
                ax.set_title('ROC-AUC для разных моделей')
                ax.set_xlim([0, 1])
                plt.tight_layout()
                st.pyplot(fig)

        # Precision-Recall кривые
        st.subheader("📊 Precision-Recall метрики")
        if any('pr_auc' in str(results.get(k, {})).lower() for k in results.keys()):
            pr_data = {}
            for model_name, model_results in results.items():
                if isinstance(model_results, dict) and 'pr_auc' in model_results:
                    pr_data[model_name] = model_results['pr_auc']

            if pr_data:
                fig, ax = plt.subplots(figsize=(10, 6))
                models = list(pr_data.keys())
                values = list(pr_data.values())
                ax.barh(models, values, color='orange', alpha=0.7)
                ax.set_xlabel('PR-AUC')
                ax.set_title('PR-AUC для разных моделей')
                ax.set_xlim([0, 1])
                plt.tight_layout()
                st.pyplot(fig)
        else:
            st.info("PR-AUC метрики не найдены в результатах")

        # Confusion Matrices
        st.subheader("📊 Матрицы ошибок (Confusion Matrices)")

        # Находим все confusion matrices
        cm_models = []
        for model_name, model_results in results.items():
            if isinstance(model_results, dict) and 'confusion_matrix' in model_results:
                cm_models.append(model_name)

        if cm_models:
            # Показываем confusion matrices для всех моделей
            cols = st.columns(min(2, len(cm_models)))
            for idx, model_name in enumerate(cm_models):
                with cols[idx % len(cols)]:
                    cm = results[model_name]['confusion_matrix']
                    if isinstance(cm, list):
                        cm_array = np.array(cm)

                        # Получаем имена классов из результатов
                        class_names = None
                        if 'classification_report' in results[model_name]:
                            report = results[model_name]['classification_report']
                            if isinstance(report, dict):
                                # Извлекаем имена классов из отчета
                                class_names = [k for k in report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]

                        if class_names is None:
                            class_names = [f"Класс {i}" for i in range(len(cm_array))]

                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                                   xticklabels=class_names, yticklabels=class_names, ax=ax)
                        ax.set_ylabel('Истинные значения')
                        ax.set_xlabel('Предсказанные значения')
                        ax.set_title(f'{model_name}')
                        plt.tight_layout()
                        st.pyplot(fig)
        else:
            st.info("Матрицы ошибок не найдены в результатах")
    
    else:
        st.info("👆 Загрузите JSON файл с результатами или выберите из доступных результатов")
        if available_results:
            st.write("**Доступные результаты:**")
            for task, files in available_results.items():
                st.write(f"- **{task}**:")
                for file in files:
                    st.write(f"  - `{os.path.basename(file)}`")

# Футер
st.markdown("---")
st.markdown("**Лабораторная работа №3: Классификация текстов**")