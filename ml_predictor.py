"""Week 8: Classical Machine Learning & Rational Decision-Making Module."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


FEATURE_COLUMNS = [
    "past_avg_grade",
    "midterm_score",
    "attendance_rate",
    "assignments_completed",
    "study_hours_planned",
    "avg_sleep_hours",
    "fatigue_index",
    "subject_difficulty",
]

FEATURE_LABELS_RU = {
    "past_avg_grade": "Средний балл ранее",
    "midterm_score": "Балл Midterm",
    "attendance_rate": "Посещаемость (%)",
    "assignments_completed": "Выполнено ДЗ (%)",
    "study_hours_planned": "Часы подготовки",
    "avg_sleep_hours": "Средний сон (ч)",
    "fatigue_index": "Индекс усталости",
    "subject_difficulty": "Сложность предмета",
}


@dataclass
class SubjectPrediction:
    """Prediction and rational study-hour recommendation for a single subject."""
    predicted_score: float
    risk_category: str
    risk_probabilities: Dict[str, float]
    score_gap: float
    expected_utility_priority: float
    recommended_sessions: int


class ExamScorePredictor:
    """Trains, evaluates, and serves Classical ML models for exam score & risk forecasting."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.best_regressor = None
        self.best_regressor_name: str = ""
        self.classifier = None
        self.regression_metrics: Dict[str, Dict[str, float]] = {}
        self.classification_metrics: Dict[str, Dict[str, float]] = {}
        self.feature_importances: Dict[str, float] = {}
        self.is_trained: bool = False

    def train_and_evaluate(self, df: pd.DataFrame) -> Dict[str, object]:
        """Train multiple classical ML models, run GridSearchCV, and collect benchmark metrics."""
        missing = [c for c in FEATURE_COLUMNS + ["final_score", "grade_category"] if c not in df.columns]
        if missing:
            raise ValueError(f"В датасете отсутствуют столбцы: {missing}")

        X = df[FEATURE_COLUMNS].copy()
        y_reg = df["final_score"].astype(float)
        y_cls = df["grade_category"].astype(str)

        X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(
            X, y_reg, y_cls, test_size=0.2, random_state=self.random_state, stratify=y_cls
        )

        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=self.random_state),
            param_grid={
                "n_estimators": [80, 140],
                "max_depth": [6, 10, None],
                "min_samples_leaf": [2, 4],
            },
            cv=4,
            scoring="r2",
            n_jobs=1,
        )

        regressors = {
            "Linear Regression": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
            "Ridge Regression": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=5.0))]),
            "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=5, random_state=self.random_state),
            "Random Forest (GridSearch)": rf_grid,
        }

        best_r2 = -np.inf
        fitted_models = {}

        for name, model in regressors.items():
            model.fit(X_train, y_reg_train)
            estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model
            preds = np.clip(estimator.predict(X_test), 0.0, 100.0)

            mae = float(mean_absolute_error(y_reg_test, preds))
            rmse = float(math.sqrt(mean_squared_error(y_reg_test, preds)))
            r2 = float(r2_score(y_reg_test, preds))
            cv_r2 = float(np.mean(cross_val_score(estimator, X_train, y_reg_train, cv=4, scoring="r2")))

            self.regression_metrics[name] = {
                "MAE": round(mae, 3),
                "RMSE": round(rmse, 3),
                "R2": round(r2, 4),
                "CV_R2": round(cv_r2, 4),
            }
            fitted_models[name] = estimator
            if r2 > best_r2:
                best_r2 = r2
                self.best_regressor = estimator
                self.best_regressor_name = name

        rf_model = fitted_models["Random Forest (GridSearch)"]
        self.feature_importances = {
            col: round(float(imp), 4)
            for col, imp in zip(FEATURE_COLUMNS, rf_model.feature_importances_)
        }

        classifiers = {
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=self.random_state)),
            ]),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=140, max_depth=8, min_samples_leaf=3, random_state=self.random_state
            ),
        }

        best_f1 = -np.inf
        for name, clf in classifiers.items():
            clf.fit(X_train, y_cls_train)
            cls_preds = clf.predict(X_test)
            acc = float(accuracy_score(y_cls_test, cls_preds))
            f1 = float(f1_score(y_cls_test, cls_preds, average="weighted"))
            self.classification_metrics[name] = {
                "Accuracy": round(acc, 4),
                "Weighted_F1": round(f1, 4),
            }
            if f1 > best_f1:
                best_f1 = f1
                self.classifier = clf

        self.is_trained = True
        return {
            "best_regressor": self.best_regressor_name,
            "regression_metrics": self.regression_metrics,
            "classification_metrics": self.classification_metrics,
            "feature_importances": self.feature_importances,
            "test_actual": y_reg_test.tolist(),
            "test_pred": np.clip(self.best_regressor.predict(X_test), 0.0, 100.0).tolist(),
        }

    @staticmethod
    def validate_features(features: Dict[str, float]) -> Dict[str, float]:
        """Validate user input ranges to prevent invalid model inference."""
        bounds = {
            "past_avg_grade": (0.0, 100.0),
            "midterm_score": (0.0, 100.0),
            "attendance_rate": (0.0, 100.0),
            "assignments_completed": (0.0, 100.0),
            "study_hours_planned": (0.0, 80.0),
            "avg_sleep_hours": (2.0, 12.0),
            "fatigue_index": (1.0, 10.0),
            "subject_difficulty": (1, 5),
        }
        cleaned = {}
        for col, (lo, hi) in bounds.items():
            if col not in features:
                raise ValueError(f"Отсутствует обязательный признак: {col}")
            val = float(features[col])
            if not (lo <= val <= hi):
                raise ValueError(f"Параметр {col}={val} выходит за допустимый диапазон [{lo}, {hi}].")
            cleaned[col] = val
        return cleaned

    def predict_subject(
        self,
        features: Dict[str, float],
        target_score: float = 85.0,
        days_until_exam: int = 5,
    ) -> SubjectPrediction:
        """Predict score, grade risk, and compute rational study session budget."""
        if not self.is_trained:
            raise RuntimeError("Модель не обучена. Сначала вызовите train_and_evaluate().")

        cleaned = self.validate_features(features)
        row_df = pd.DataFrame([cleaned])[FEATURE_COLUMNS]

        pred_score = float(np.clip(self.best_regressor.predict(row_df)[0], 0.0, 100.0))
        probs_arr = self.classifier.predict_proba(row_df)[0]
        probs = {
            str(cls): round(float(p), 3)
            for cls, p in zip(self.classifier.classes_, probs_arr)
        }
        pred_cat = str(self.classifier.predict(row_df)[0])

        score_gap = max(0.0, target_score - pred_score)
        p_high_risk = probs.get("High Risk", 0.0)
        urgency = 1.0 / max(1, days_until_exam)
        eu_priority = round(
            0.04 * score_gap + 2.5 * p_high_risk + 0.35 * cleaned["subject_difficulty"] + 3.0 * urgency,
            3,
        )

        base_sessions = 1 + int(round(score_gap / 8.0)) + (1 if cleaned["subject_difficulty"] >= 4 else 0)
        if p_high_risk >= 0.4:
            base_sessions += 1
        rec_sessions = max(1, min(6, base_sessions))

        return SubjectPrediction(
            predicted_score=round(pred_score, 1),
            risk_category=pred_cat,
            risk_probabilities=probs,
            score_gap=round(score_gap, 1),
            expected_utility_priority=eu_priority,
            recommended_sessions=rec_sessions,
        )
