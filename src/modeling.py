import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier

from src.config import AGE_GROUPS, DATASET_PATH, METRICS_PATH, MODEL_BUNDLE_PATH
from src.preprocessing import build_vocabulary, feature_names, load_cases, vectorize_symptoms


def train_and_save_models(
    dataset_path: Path = DATASET_PATH,
    model_path: Path = MODEL_BUNDLE_PATH,
    metrics_path: Path = METRICS_PATH,
    random_state: int = 42,
) -> dict:
    cases = load_cases(dataset_path)
    vocabulary = build_vocabulary(cases)
    X = vectorize_symptoms(cases["symptom_list"], vocabulary, AGE_GROUPS, cases["age_group"])
    y = cases["disease"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=random_state,
        stratify=y,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    candidates = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Complement Naive Bayes": ComplementNB(),
        "Logistic Regression Tuned": GridSearchCV(
            LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state),
            {"C": [0.4, 1.0, 2.5]},
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1,
        ),
        "Random Forest Tuned": GridSearchCV(
            RandomForestClassifier(random_state=random_state, n_jobs=-1),
            {"n_estimators": [80, 140], "max_depth": [6, None], "min_samples_leaf": [1, 2]},
            scoring="f1_weighted",
            cv=cv,
            n_jobs=-1,
        ),
        "Neural Network MLP": MLPClassifier(
            hidden_layer_sizes=(48, 24),
            activation="relu",
            alpha=0.0008,
            learning_rate_init=0.01,
            max_iter=450,
            early_stopping=False,
            random_state=random_state,
        ),
    }

    fitted_models = {}
    metrics = {}
    for name, estimator in candidates.items():
        estimator.fit(X_train, y_train)
        fitted = estimator.best_estimator_ if hasattr(estimator, "best_estimator_") else estimator
        predictions = fitted.predict(X_test)
        metrics[name] = {
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "f1_weighted": round(float(f1_score(y_test, predictions, average="weighted")), 4),
            "best_params": getattr(estimator, "best_params_", None),
        }
        fitted_models[name] = fitted

    best_model_name = max(metrics, key=lambda item: (metrics[item]["f1_weighted"], metrics[item]["accuracy"]))
    best_model = fitted_models[best_model_name]
    y_pred = best_model.predict(X_test)

    output_metrics = {
        "best_model": best_model_name,
        "feature_count": int(X.shape[1]),
        "training_cases": int(len(cases)),
        "class_count": int(len(np.unique(y))),
        "models": metrics,
        "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
        "feature_names": feature_names(vocabulary, AGE_GROUPS),
    }

    bundle = {
        "best_model_name": best_model_name,
        "models": {best_model_name: best_model},
        "vocabulary": vocabulary,
        "age_groups": AGE_GROUPS,
        "labels": sorted(set(y)),
        "metrics": output_metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics_path.write_text(json.dumps(output_metrics, indent=2), encoding="utf-8")
    return output_metrics


def load_model_bundle(model_path: Path = MODEL_BUNDLE_PATH) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run scripts/train_models.py first."
        )
    return joblib.load(model_path)


def predict_diseases(symptoms: list[str], age_group: str = "adult", bundle: dict | None = None) -> list[dict]:
    bundle = bundle or load_model_bundle()
    model = bundle["models"][bundle["best_model_name"]]
    X = vectorize_symptoms([symptoms], bundle["vocabulary"], bundle["age_groups"], [age_group])

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[0]
        classes = model.classes_
    else:
        predicted = model.predict(X)[0]
        classes = np.asarray(bundle["labels"])
        probabilities = np.asarray([1.0 if label == predicted else 0.0 for label in classes])

    ranked = sorted(
        [
            {"disease": str(label), "confidence": round(float(probability), 4)}
            for label, probability in zip(classes, probabilities)
        ],
        key=lambda row: row["confidence"],
        reverse=True,
    )
    return ranked
