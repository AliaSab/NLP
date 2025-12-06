"""
Модуль для классических методов классификации.
Этап 3: Реализация классических ML-моделей.
"""

import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

# Градиентный бустинг
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# AutoML
try:
    import autosklearn.classification
    AUTOSKLEARN_AVAILABLE = True
except ImportError:
    AUTOSKLEARN_AVAILABLE = False

try:
    from tpot import TPOTClassifier
    TPOT_AVAILABLE = True
except ImportError:
    TPOT_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClassicalClassifier:
    """Класс для обучения и оценки классических моделей."""
    
    def __init__(self, task_type: str = 'binary'):
        """
        Args:
            task_type: 'binary', 'multiclass', 'multilabel'
        """
        self.task_type = task_type
        self.models = {}
        self.results = {}
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                 penalty: str = 'l2', C: float = 1.0, max_iter: int = 1000) -> LogisticRegression:
        """Обучение логистической регрессии."""
        logger.info(f"Обучение Logistic Regression (penalty={penalty}, C={C})")
        
        if self.task_type == 'multilabel':
            from sklearn.multioutput import MultiOutputClassifier
            model = MultiOutputClassifier(
                LogisticRegression(penalty=penalty, C=C, max_iter=max_iter, random_state=42)
            )
        else:
            model = LogisticRegression(
                penalty=penalty, C=C, max_iter=max_iter, random_state=42, class_weight='balanced'
            )
        
        model.fit(X_train, y_train)
        self.models['logistic_regression'] = model
        return model
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray,
                  kernel: str = 'linear', C: float = 1.0) -> SVC:
        """Обучение SVM."""
        logger.info(f"Обучение SVM (kernel={kernel}, C={C})")
        
        base_model = SVC(
            kernel=kernel, C=C, probability=True, random_state=42, class_weight='balanced'
        )
        
        if self.task_type == 'multilabel':
            from sklearn.multioutput import MultiOutputClassifier
            model = MultiOutputClassifier(base_model)
        else:
            model = base_model
        
        model.fit(X_train, y_train)
        self.models['svm'] = model
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           n_estimators: int = 100, max_depth: Optional[int] = None) -> RandomForestClassifier:
        """Обучение случайного леса."""
        logger.info(f"Обучение Random Forest (n_estimators={n_estimators})")
        
        base_model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42, class_weight='balanced'
        )
        
        if self.task_type == 'multilabel':
            from sklearn.multioutput import MultiOutputClassifier
            model = MultiOutputClassifier(base_model)
        else:
            model = base_model
        
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     n_estimators: int = 100, max_depth: int = 6) -> Optional[Any]:
        """Обучение XGBoost."""
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost не доступен")
            return None
        
        logger.info(f"Обучение XGBoost (n_estimators={n_estimators})")
        
        base_model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        
        if self.task_type == 'multilabel':
            from sklearn.multioutput import MultiOutputClassifier
            model = MultiOutputClassifier(base_model)
        else:
            model = base_model
        
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        return model
    
    def train_catboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      iterations: int = 100, depth: int = 6) -> Optional[Any]:
        """Обучение CatBoost."""
        if not CATBOOST_AVAILABLE:
            logger.warning("CatBoost не доступен")
            return None
        
        logger.info(f"Обучение CatBoost (iterations={iterations})")
        
        base_model = CatBoostClassifier(
            iterations=iterations, depth=depth, random_state=42, verbose=False
        )
        
        if self.task_type == 'multilabel':
            from sklearn.multioutput import MultiOutputClassifier
            model = MultiOutputClassifier(base_model)
        else:
            model = base_model
        
        model.fit(X_train, y_train)
        self.models['catboost'] = model
        return model
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray,
                      n_estimators: int = 100, max_depth: int = 6) -> Optional[Any]:
        """Обучение LightGBM."""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM не доступен")
            return None
        
        logger.info(f"Обучение LightGBM (n_estimators={n_estimators})")
        
        base_model = lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42, class_weight='balanced',
            verbosity=-1  # Подавляем предупреждения
        )
        
        if self.task_type == 'multilabel':
            from sklearn.multioutput import MultiOutputClassifier
            model = MultiOutputClassifier(base_model)
        else:
            model = base_model
        
        model.fit(X_train, y_train)
        self.models['lightgbm'] = model
        return model
    
    def train_bagging(self, X_train: np.ndarray, y_train: np.ndarray,
                     base_estimator=None, n_estimators: int = 10) -> BaggingClassifier:
        """Обучение Bagging."""
        logger.info(f"Обучение Bagging (n_estimators={n_estimators})")
        
        if base_estimator is None:
            base_estimator = DecisionTreeClassifier(random_state=42)
        
        # В scikit-learn >= 1.2.0 параметр base_estimator заменен на estimator
        try:
            # Попытка использовать новый параметр (scikit-learn >= 1.2.0)
            bagging_model = BaggingClassifier(
                estimator=base_estimator, n_estimators=n_estimators, random_state=42
            )
        except TypeError:
            # Fallback для старых версий (scikit-learn < 1.2.0)
            bagging_model = BaggingClassifier(
                base_estimator=base_estimator, n_estimators=n_estimators, random_state=42
            )
        
        if self.task_type == 'multilabel':
            from sklearn.multioutput import MultiOutputClassifier
            model = MultiOutputClassifier(bagging_model)
        else:
            model = bagging_model
        
        model.fit(X_train, y_train)
        self.models['bagging'] = model
        return model
    
    def train_stacking(self, X_train: np.ndarray, y_train: np.ndarray) -> VotingClassifier:
        """Обучение Stacking/Blending с Voting."""
        logger.info("Обучение Stacking/Voting")
        
        # Базовые модели
        estimators = []
        
        # SVM
        svm = SVC(kernel='linear', probability=True, random_state=42, class_weight='balanced')
        if self.task_type == 'multilabel':
            from sklearn.multioutput import MultiOutputClassifier
            svm = MultiOutputClassifier(svm)
        estimators.append(('svm', svm))
        
        # Logistic Regression
        lr = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
        if self.task_type == 'multilabel':
            from sklearn.multioutput import MultiOutputClassifier
            lr = MultiOutputClassifier(lr)
        estimators.append(('lr', lr))
        
        # CatBoost (если доступен)
        if CATBOOST_AVAILABLE:
            cb = CatBoostClassifier(iterations=50, random_state=42, verbose=False)
            if self.task_type == 'multilabel':
                from sklearn.multioutput import MultiOutputClassifier
                cb = MultiOutputClassifier(cb)
            estimators.append(('catboost', cb))
        else:
            # Random Forest как замена
            rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced')
            if self.task_type == 'multilabel':
                from sklearn.multioutput import MultiOutputClassifier
                rf = MultiOutputClassifier(rf)
            estimators.append(('rf', rf))
        
        # Voting Classifier
        # Для multilabel VotingClassifier не поддерживается напрямую, используем только первую модель
        if self.task_type == 'multilabel':
            logger.warning("VotingClassifier не поддерживает multilabel, используется только первая модель")
            model = estimators[0][1]  # Берем первую модель
        else:
            model = VotingClassifier(estimators=estimators, voting='soft')
        
        model.fit(X_train, y_train)
        self.models['stacking'] = model
        return model
    
    def train_autosklearn(self, X_train: np.ndarray, y_train: np.ndarray,
                         time_left_for_this_task: int = 300) -> Optional[Any]:
        """Обучение AutoSklearn."""
        if not AUTOSKLEARN_AVAILABLE:
            logger.warning("AutoSklearn не доступен")
            return None
        
        logger.info(f"Обучение AutoSklearn (time_limit={time_left_for_this_task}s)")
        
        model = autosklearn.classification.AutoSklearnClassifier(
            time_left_for_this_task=time_left_for_this_task,
            memory_limit=3072,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['autosklearn'] = model
        return model
    
    def train_tpot(self, X_train: np.ndarray, y_train: np.ndarray,
                  generations: int = 5, population_size: int = 20) -> Optional[Any]:
        """Обучение TPOT."""
        if not TPOT_AVAILABLE:
            logger.warning("TPOT не доступен")
            return None
        
        logger.info(f"Обучение TPOT (generations={generations})")
        
        model = TPOTClassifier(
            generations=generations, population_size=population_size, random_state=42, verbosity=0
        )
        model.fit(X_train, y_train)
        self.models['tpot'] = model
        return model
    
    def evaluate(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                model_name: str = 'model') -> Dict[str, Any]:
        """Оценка модели."""
        y_pred = model.predict(X_test)
        
        # Базовые метрики
        accuracy = accuracy_score(y_test, y_pred)
        
        if self.task_type == 'binary':
            precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
            recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            
            # ROC-AUC
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    roc_auc = roc_auc_score(y_test, y_proba)
                else:
                    roc_auc = None
            except:
                roc_auc = None
            
            # PR-AUC
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    pr_auc = average_precision_score(y_test, y_proba)
                else:
                    pr_auc = None
            except:
                pr_auc = None
            
            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc
            }
            
        elif self.task_type == 'multiclass':
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            results = {
                'accuracy': accuracy,
                'precision_macro': precision,
                'recall_macro': recall,
                'f1_score_macro': f1,
                'precision_micro': precision_score(y_test, y_pred, average='micro', zero_division=0),
                'recall_micro': recall_score(y_test, y_pred, average='micro', zero_division=0),
                'f1_score_micro': f1_score(y_test, y_pred, average='micro', zero_division=0)
            }
        
        else:  # multilabel
            # Для multilabel используем subset accuracy (точное совпадение всех меток)
            # и метрики по каждому классу
            precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            # Subset accuracy (точное совпадение всех меток)
            subset_accuracy = accuracy_score(y_test, y_pred)
            
            results = {
                'accuracy': subset_accuracy,  # Subset accuracy для multilabel
                'precision_macro': precision,
                'recall_macro': recall,
                'f1_score_macro': f1,
                'precision_micro': precision_score(y_test, y_pred, average='micro', zero_division=0),
                'recall_micro': recall_score(y_test, y_pred, average='micro', zero_division=0),
                'f1_score_micro': f1_score(y_test, y_pred, average='micro', zero_division=0)
            }
        
        # Confusion matrix (не применима для multilabel)
        if self.task_type != 'multilabel':
            cm = confusion_matrix(y_test, y_pred)
            results['confusion_matrix'] = cm.tolist()
        else:
            # Для multilabel confusion matrix не строится
            results['confusion_matrix'] = None
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results['classification_report'] = report
        
        self.results[model_name] = results
        return results
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Обучение всех доступных моделей."""
        logger.info("Обучение всех классических моделей...")
        
        # Базовые модели
        self.train_logistic_regression(X_train, y_train)
        self.train_svm(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        
        # Градиентный бустинг
        if XGBOOST_AVAILABLE:
            self.train_xgboost(X_train, y_train)
        if CATBOOST_AVAILABLE:
            self.train_catboost(X_train, y_train)
        if LIGHTGBM_AVAILABLE:
            self.train_lightgbm(X_train, y_train)
        
        # Ансамбли
        self.train_bagging(X_train, y_train)
        self.train_stacking(X_train, y_train)
        
        # Оценка всех моделей
        all_results = {}
        for model_name, model in self.models.items():
            logger.info(f"Оценка модели: {model_name}")
            results = self.evaluate(model, X_test, y_test, model_name)
            all_results[model_name] = results
        
        return all_results
    
    def save_results(self, file_path: str):
        """Сохранение результатов в JSON."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"Результаты сохранены в {file_path}")

