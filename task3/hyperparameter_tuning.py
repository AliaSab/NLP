"""
Модуль для настройки гиперпараметров и оценки моделей.
Этап 6: Настройка и оценка моделей.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit, GroupKFold
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, make_scorer
)
import warnings
warnings.filterwarnings('ignore')

# Bayesian Optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    from hyperopt import fmin, tpe, hp, Trials
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CrossValidator:
    """Класс для кросс-валидации."""
    
    def __init__(self, cv_type: str = 'stratified', n_splits: int = 5, random_state: int = 42):
        """
        Args:
            cv_type: 'stratified', 'time_series', 'group'
            n_splits: Количество фолдов
        """
        self.cv_type = cv_type
        self.n_splits = n_splits
        self.random_state = random_state
    
    def get_cv(self, y: Optional[np.ndarray] = None, groups: Optional[np.ndarray] = None):
        """Получение объекта кросс-валидации."""
        if self.cv_type == 'stratified':
            if y is None:
                raise ValueError("y требуется для StratifiedKFold")
            return StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        elif self.cv_type == 'time_series':
            return TimeSeriesSplit(n_splits=self.n_splits)
        elif self.cv_type == 'group':
            if groups is None:
                raise ValueError("groups требуется для GroupKFold")
            return GroupKFold(n_splits=self.n_splits)
        else:
            raise ValueError(f"Неизвестный тип CV: {self.cv_type}")


class HyperparameterTuner:
    """Класс для настройки гиперпараметров."""
    
    def __init__(self, task_type: str = 'binary', cv_type: str = 'stratified', n_splits: int = 5):
        self.task_type = task_type
        self.cv = CrossValidator(cv_type, n_splits)
        self.best_params = {}
        self.best_scores = {}
    
    def grid_search(self, model, param_grid: Dict, X: np.ndarray, y: np.ndarray,
                   scoring: str = None, n_jobs: int = -1) -> Any:
        """Grid Search - полный перебор."""
        logger.info("Grid Search...")
        
        if scoring is None:
            if self.task_type == 'binary':
                scoring = 'roc_auc'
            else:
                scoring = 'f1_macro'
        
        cv = self.cv.get_cv(y=y)
        
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        self.best_params['grid_search'] = grid_search.best_params_
        self.best_scores['grid_search'] = grid_search.best_score_
        
        logger.info(f"Лучшие параметры: {grid_search.best_params_}")
        logger.info(f"Лучший score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def random_search(self, model, param_distributions: Dict, X: np.ndarray, y: np.ndarray,
                     n_iter: int = 100, scoring: str = None, n_jobs: int = -1) -> Any:
        """Random Search - случайный поиск."""
        logger.info(f"Random Search (n_iter={n_iter})...")
        
        if scoring is None:
            if self.task_type == 'binary':
                scoring = 'roc_auc'
            else:
                scoring = 'f1_macro'
        
        cv = self.cv.get_cv(y=y)
        
        random_search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=42,
            verbose=1
        )
        
        random_search.fit(X, y)
        
        self.best_params['random_search'] = random_search.best_params_
        self.best_scores['random_search'] = random_search.best_score_
        
        logger.info(f"Лучшие параметры: {random_search.best_params_}")
        logger.info(f"Лучший score: {random_search.best_score_:.4f}")
        
        return random_search.best_estimator_
    
    def optuna_optimization(self, model_class, param_space: Dict, X: np.ndarray, y: np.ndarray,
                           n_trials: int = 100, scoring: str = None) -> Any:
        """Bayesian Optimization с Optuna."""
        if not OPTUNA_AVAILABLE:
            logger.warning("Optuna не доступен")
            return None
        
        logger.info(f"Optuna Optimization (n_trials={n_trials})...")
        
        if scoring is None:
            if self.task_type == 'binary':
                scoring = 'roc_auc'
            else:
                scoring = 'f1_macro'
        
        cv = self.cv.get_cv(y=y)
        
        def objective(trial):
            # Создание параметров из пространства поиска
            params = {}
            for param_name, param_type in param_space.items():
                if param_type['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_type['choices'])
                elif param_type['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_type['low'], param_type['high'])
                elif param_type['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_type['low'], param_type['high'], log=param_type.get('log', False))
            
            # Создание модели с параметрами
            model = model_class(**params)
            
            # Кросс-валидация
            scores = []
            for train_idx, val_idx in cv.split(X, y):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                
                model.fit(X_train_cv, y_train_cv)
                y_pred = model.predict(X_val_cv)
                
                if scoring == 'roc_auc' and self.task_type == 'binary':
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_val_cv)[:, 1]
                        score = roc_auc_score(y_val_cv, y_proba)
                    else:
                        score = accuracy_score(y_val_cv, y_pred)
                elif scoring == 'f1_macro':
                    score = f1_score(y_val_cv, y_pred, average='macro')
                else:
                    score = accuracy_score(y_val_cv, y_pred)
                
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        self.best_params['optuna'] = best_params
        self.best_scores['optuna'] = study.best_value
        
        logger.info(f"Лучшие параметры: {best_params}")
        logger.info(f"Лучший score: {study.best_value:.4f}")
        
        # Создание финальной модели
        model = model_class(**best_params)
        model.fit(X, y)
        
        return model
    
    def hyperopt_optimization(self, model_class, param_space: Dict, X: np.ndarray, y: np.ndarray,
                            max_evals: int = 100, scoring: str = None) -> Any:
        """Bayesian Optimization с Hyperopt."""
        if not HYPEROPT_AVAILABLE:
            logger.warning("Hyperopt не доступен")
            return None
        
        logger.info(f"Hyperopt Optimization (max_evals={max_evals})...")
        
        if scoring is None:
            if self.task_type == 'binary':
                scoring = 'roc_auc'
            else:
                scoring = 'f1_macro'
        
        cv = self.cv.get_cv(y=y)
        
        def objective(params):
            model = model_class(**params)
            
            scores = []
            for train_idx, val_idx in cv.split(X, y):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv, y_val_cv = y[train_idx], y[val_idx]
                
                model.fit(X_train_cv, y_train_cv)
                y_pred = model.predict(X_val_cv)
                
                if scoring == 'roc_auc' and self.task_type == 'binary':
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_val_cv)[:, 1]
                        score = roc_auc_score(y_val_cv, y_proba)
                    else:
                        score = accuracy_score(y_val_cv, y_pred)
                elif scoring == 'f1_macro':
                    score = f1_score(y_val_cv, y_pred, average='macro')
                else:
                    score = accuracy_score(y_val_cv, y_pred)
                
                scores.append(score)
            
            return -np.mean(scores)  # Hyperopt минимизирует
        
        trials = Trials()
        best = fmin(
            fn=objective,
            space=param_space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials
        )
        
        self.best_params['hyperopt'] = best
        self.best_scores['hyperopt'] = -trials.best_trial['result']['loss']
        
        logger.info(f"Лучшие параметры: {best}")
        logger.info(f"Лучший score: {-trials.best_trial['result']['loss']:.4f}")
        
        model = model_class(**best)
        model.fit(X, y)
        
        return model


class ModelEvaluator:
    """Класс для всесторонней оценки моделей."""
    
    def __init__(self, task_type: str = 'binary'):
        self.task_type = task_type
        self.results = {}
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = 'model') -> Dict[str, Any]:
        """Комплексная оценка модели."""
        y_pred = model.predict(X_test)
        
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
        }
        
        if self.task_type == 'binary':
            results.update({
                'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
                'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0),
            })
            
            # ROC-AUC
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    results['roc_auc'] = roc_auc_score(y_test, y_proba)
                    results['pr_auc'] = average_precision_score(y_test, y_proba)
            except:
                pass
        
        elif self.task_type == 'multiclass':
            results.update({
                'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'f1_score_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
                'precision_micro': precision_score(y_test, y_pred, average='micro', zero_division=0),
                'recall_micro': recall_score(y_test, y_pred, average='micro', zero_division=0),
                'f1_score_micro': f1_score(y_test, y_pred, average='micro', zero_division=0),
            })
        
        else:  # multilabel
            results.update({
                'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
                'f1_score_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
            })
        
        self.results[model_name] = results
        return results
    
    def cross_validate(self, model: Any, X: np.ndarray, y: np.ndarray,
                      cv: Any, scoring: str = None) -> Dict[str, Any]:
        """Кросс-валидация модели."""
        from sklearn.model_selection import cross_val_score
        
        if scoring is None:
            if self.task_type == 'binary':
                scoring = 'roc_auc'
            else:
                scoring = 'f1_macro'
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'scores': scores.tolist()
        }
    
    def save_results(self, file_path: str):
        """Сохранение результатов."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"Результаты сохранены в {file_path}")

