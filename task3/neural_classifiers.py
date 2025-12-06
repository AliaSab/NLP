"""
Модуль для нейросетевых и трансформерных моделей классификации.
Этап 4: Реализация нейросетевых и трансформерных методов.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, optimizers, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

# PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    PYTORCH_AVAILABLE = True
    Dataset = Dataset  # Для использования в условном определении класса
except ImportError:
    PYTORCH_AVAILABLE = False
    Dataset = None  # Заглушка

# Transformers
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        Trainer, TrainingArguments, DataCollatorWithPadding
    )
    from datasets import Dataset as HFDataset
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# TextDataset только если PyTorch доступен
if PYTORCH_AVAILABLE and Dataset is not None:
    class TextDataset(Dataset):
        """Dataset для PyTorch."""
        
        def __init__(self, texts: List[str], labels: List[Any], tokenizer, max_length: int = 512):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            label = self.labels[idx]
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }
else:
    # Заглушка если PyTorch не установлен
    class TextDataset:
        """Заглушка для TextDataset когда PyTorch не установлен."""
        pass


class NeuralClassifier:
    """Класс для обучения нейросетевых моделей."""
    
    def __init__(self, task_type: str = 'binary', num_classes: int = 2):
        self.task_type = task_type
        self.num_classes = num_classes
        self.models = {}
        self.results = {}
    
    def build_mlp(self, input_dim: int, hidden_dims: List[int] = [128, 64],
                 dropout_rate: float = 0.5) -> Optional[Any]:
        """Построение многослойного персептрона."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow не доступен")
            return None
        
        logger.info(f"Построение MLP (input_dim={input_dim}, hidden_dims={hidden_dims})")
        
        model = models.Sequential()
        model.add(layers.Input(shape=(input_dim,)))
        
        for dim in hidden_dims:
            model.add(layers.Dense(dim, activation='relu'))
            model.add(layers.Dropout(dropout_rate))
        
        if self.task_type == 'binary':
            model.add(layers.Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy']
        )
        
        self.models['mlp'] = model
        return model
    
    def build_cnn(self, vocab_size: int, embedding_dim: int = 100, 
                 max_length: int = 500, num_filters: int = 128,
                 filter_sizes: List[int] = [3, 4, 5], dropout_rate: float = 0.5) -> Optional[Any]:
        """Построение CNN для текста (архитектура Kim CNN)."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow не доступен")
            return None
        
        logger.info("Построение CNN (Kim architecture)")
        
        inputs = layers.Input(shape=(max_length,))
        embedding = layers.Embedding(vocab_size, embedding_dim)(inputs)
        
        conv_blocks = []
        for filter_size in filter_sizes:
            conv = layers.Conv1D(filters=num_filters, kernel_size=filter_size, activation='relu')(embedding)
            conv = layers.GlobalMaxPooling1D()(conv)
            conv_blocks.append(conv)
        
        concat = layers.Concatenate()(conv_blocks)
        dropout = layers.Dropout(dropout_rate)(concat)
        
        if self.task_type == 'binary':
            outputs = layers.Dense(1, activation='sigmoid')(dropout)
            loss = 'binary_crossentropy'
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(dropout)
            loss = 'sparse_categorical_crossentropy'
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy']
        )
        
        self.models['cnn'] = model
        return model
    
    def build_lstm(self, vocab_size: int, embedding_dim: int = 100,
                  max_length: int = 500, lstm_units: int = 128,
                  bidirectional: bool = True, dropout_rate: float = 0.5) -> Optional[Any]:
        """Построение LSTM/GRU модели."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow не доступен")
            return None
        
        logger.info(f"Построение {'Bi' if bidirectional else ''}LSTM")
        
        inputs = layers.Input(shape=(max_length,))
        embedding = layers.Embedding(vocab_size, embedding_dim)(inputs)
        
        if bidirectional:
            lstm = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(embedding)
            lstm = layers.Bidirectional(layers.LSTM(lstm_units // 2))(lstm)
        else:
            lstm = layers.LSTM(lstm_units, return_sequences=True)(embedding)
            lstm = layers.LSTM(lstm_units // 2)(lstm)
        
        dropout = layers.Dropout(dropout_rate)(lstm)
        
        if self.task_type == 'binary':
            outputs = layers.Dense(1, activation='sigmoid')(dropout)
            loss = 'binary_crossentropy'
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(dropout)
            loss = 'sparse_categorical_crossentropy'
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy']
        )
        
        self.models['lstm'] = model
        return model
    
    def build_cnn_lstm(self, vocab_size: int, embedding_dim: int = 100,
                      max_length: int = 500, num_filters: int = 64,
                      lstm_units: int = 128, dropout_rate: float = 0.5) -> Optional[Any]:
        """Построение гибридной CNN+LSTM модели."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow не доступен")
            return None
        
        logger.info("Построение CNN+LSTM")
        
        inputs = layers.Input(shape=(max_length,))
        embedding = layers.Embedding(vocab_size, embedding_dim)(inputs)
        
        # CNN для извлечения локальных признаков
        conv = layers.Conv1D(filters=num_filters, kernel_size=3, activation='relu')(embedding)
        conv = layers.MaxPooling1D(pool_size=2)(conv)
        
        # LSTM для анализа последовательностей
        lstm = layers.LSTM(lstm_units)(conv)
        dropout = layers.Dropout(dropout_rate)(lstm)
        
        if self.task_type == 'binary':
            outputs = layers.Dense(1, activation='sigmoid')(dropout)
            loss = 'binary_crossentropy'
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(dropout)
            loss = 'sparse_categorical_crossentropy'
        
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=loss,
            metrics=['accuracy']
        )
        
        self.models['cnn_lstm'] = model
        return model
    
    def train_keras_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         epochs: int = 10, batch_size: int = 32,
                         model_name: str = 'model') -> Any:
        """Обучение Keras модели."""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        logger.info(f"Обучение {model_name} (epochs={epochs})")
        
        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def load_bert_model(self, model_name: str = 'DeepPavlov/rubert-base-cased') -> Optional[Any]:
        """Загрузка предобученной BERT модели."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers не доступен")
            return None
        
        logger.info(f"Загрузка BERT модели: {model_name}")
        
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self.num_classes
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            self.models['bert'] = model
            self.models['bert_tokenizer'] = tokenizer
            return model, tokenizer
        except Exception as e:
            logger.error(f"Ошибка загрузки BERT: {e}")
            return None
    
    def train_bert(self, texts: List[str], labels: List[Any],
                  model_name: str = 'DeepPavlov/rubert-base-cased',
                  epochs: int = 3, batch_size: int = 16) -> Optional[Any]:
        """Fine-tuning BERT модели."""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers не доступен")
            return None
        
        logger.info(f"Fine-tuning BERT: {model_name}")
        
        # Загрузка модели и токенизатора
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_classes
        )
        
        # Подготовка данных
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=512
            )
        
        dataset = HFDataset.from_dict({'text': texts, 'labels': labels})
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./bert_results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_strategy='epoch',
            evaluation_strategy='epoch'
        )
        
        # Метрики
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return {
                'accuracy': accuracy_score(labels, predictions),
                'f1': f1_score(labels, predictions, average='macro')
            }
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset,
            compute_metrics=compute_metrics
        )
        
        # Обучение
        trainer.train()
        
        self.models['bert'] = model
        self.models['bert_tokenizer'] = tokenizer
        
        return model
    
    def evaluate_neural_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                             model_name: str = 'model') -> Dict[str, Any]:
        """Оценка нейросетевой модели."""
        if hasattr(model, 'predict'):
            y_pred_proba = model.predict(X_test, verbose=0)
            
            if self.task_type == 'binary':
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            else:
                y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            # Для BERT через Trainer
            y_pred = model.predict(X_test)
            y_pred = np.argmax(y_pred.predictions, axis=1)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        if self.task_type == 'binary':
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
        else:
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        self.results[model_name] = results
        return results

