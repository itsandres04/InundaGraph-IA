"""Clasificación de criticidad de intersecciones con un árbol de decisión.

Criterios de etiquetado (documentados, explícitos y verificables — ABP §9.6):

    Nivel de criticidad de un nodo bajo un escenario:
      - ALTA: el nodo queda aislado (pierde todo acceso a los puntos
        seguros) o pierde al menos una arista clave manteniendo grado bajo.
      - MEDIA: el nodo pierde aristas o su intermediación cae de forma
        relevante, pero conserva acceso a algún punto seguro.
      - BAJA: bloqueo nulo y sin signos de pérdida de conectividad.

    Definiciones:
      pérdida_de_intermediación = betweenness_base - betweenness_escenario

IMPORTANTE (ABP §9.6 y §10.6): como las etiquetas provienen de reglas del
equipo, el árbol reproduce esas reglas; su exactitud no demuestra capacidad
de predecir inundaciones reales. Se usa para estudiar clasificación
explicable con DecisionTreeClassifier de scikit-learn.
"""
from __future__ import annotations

import io
from statistics import mean, pstdev

import pandas as pd
from sklearn import tree as sk_tree
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier

FEATURE_COLUMNS = [
    "degree",
    "scenario_degree",
    "betweenness",
    "scenario_betweenness",
    "affected_edges",
    "blocked_edges",
    "isolated",
]

LABELS = {"low": "baja", "medium": "media", "high": "alta"}


def explain_node_criticality(record: dict) -> tuple[str, str]:
    """Devuelve la etiqueta basada en reglas y el criterio que la activa."""
    if record["isolated"]:
        return LABELS["high"], "Perdió acceso a todos los destinos configurados."
    if record["blocked_edges"] >= 2 and record["scenario_degree"] <= 2:
        return (
            LABELS["high"],
            "Tiene al menos dos vías cerradas y conserva como máximo dos conexiones.",
        )
    if record["blocked_edges"] >= 1:
        return LABELS["medium"], "Tiene al menos una vía adyacente cerrada."
    if record["affected_edges"] >= 1:
        return LABELS["medium"], "Tiene al menos una vía adyacente afectada."
    if record["scenario_betweenness"] < record["betweenness"] * 0.5 and record["betweenness"] > 0:
        return (
            LABELS["medium"],
            "Su intermediación estimada disminuyó al menos un 50 %.",
        )
    return LABELS["low"], "No cumple los criterios de criticidad alta o media."


def label_node(record: dict) -> str:
    """Etiqueta de criticidad según las reglas documentadas del equipo."""
    return explain_node_criticality(record)[0]


def build_features(indicators: dict) -> pd.DataFrame:
    """Arma el conjunto de datos (características + etiqueta) por escenario."""
    rows = []
    for record in indicators["nodes"]:
        rows.append(
            {
                "node": record["node"],
                "degree": record["degree"],
                "scenario_degree": record["scenario_degree"],
                "betweenness": record["betweenness"],
                "scenario_betweenness": record["scenario_betweenness"],
                "affected_edges": record["affected_edges"],
                "blocked_edges": record["blocked_edges"],
                "isolated": int(record["isolated"]),
                "criticality": label_node(record),
            }
        )
    return pd.DataFrame(rows)


def evaluate_by_scenario(
    features: pd.DataFrame,
    scenario_column: str = "scenario",
    depth: int = 4,
) -> dict:
    """Evalúa dejando fuera un escenario completo en cada iteración.

    Las filas del mismo escenario nunca se reparten entre entrenamiento y
    prueba. Las métricas se promedian entre escenarios, no entre filas.
    """
    if (
        features.empty
        or scenario_column not in features
        or features[scenario_column].nunique() < 3
        or features["criticality"].nunique() < 2
    ):
        return {}

    X = features[FEATURE_COLUMNS].astype("float64").to_numpy()
    y = features["criticality"].to_numpy(dtype=str)
    groups = features[scenario_column].to_numpy()
    labels = sorted(features["criticality"].unique())
    fold_metrics = []
    skipped_folds = 0

    for train_indices, test_indices in LeaveOneGroupOut().split(X, y, groups):
        y_train = y[train_indices]
        if len(set(y_train)) < 2:
            skipped_folds += 1
            continue

        model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=7,
        )
        model.fit(X[train_indices], y_train)
        predictions = model.predict(X[test_indices])
        y_test = y[test_indices]
        fold_metrics.append(
            {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(y_test, predictions)
                ),
                "precision": float(
                    precision_score(
                        y_test, predictions, labels=labels, average="macro",
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        y_test, predictions, labels=labels, average="macro",
                        zero_division=0,
                    )
                ),
                "f1": float(
                    f1_score(
                        y_test, predictions, labels=labels, average="macro",
                        zero_division=0,
                    )
                ),
            }
        )

    if not fold_metrics:
        return {
            "scenario_count": int(features[scenario_column].nunique()),
            "folds": 0,
            "skipped_folds": skipped_folds,
        }

    metrics = {
        key: round(mean(fold[key] for fold in fold_metrics), 3)
        for key in fold_metrics[0]
    }
    metrics.update(
        {
            f"{key}_std": round(pstdev(fold[key] for fold in fold_metrics), 3)
            for key in fold_metrics[0]
        }
    )
    metrics.update(
        {
            "scenario_count": int(features[scenario_column].nunique()),
            "folds": len(fold_metrics),
            "skipped_folds": skipped_folds,
        }
    )
    return metrics


def train_decision_tree(
    features: pd.DataFrame,
    depth: int = 4,
    scenario_column: str = "scenario",
):
    """Ajusta el árbol con todos los registros y evalúa por escenario.

    Devuelve (modelo final, reglas, métricas de validación agrupada). El
    modelo completo es explicativo; las métricas proceden de modelos
    independientes entrenados sin el escenario evaluado.
    """
    metrics = evaluate_by_scenario(features, scenario_column, depth)
    if features.empty or features["criticality"].nunique() < 2:
        return None, "Conjunto de datos insuficiente para entrenar el árbol.", metrics

    X = features[FEATURE_COLUMNS].astype("float64").to_numpy()
    y = features["criticality"].to_numpy(dtype=str)
    model = DecisionTreeClassifier(
        max_depth=depth,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=7,
    )
    model.fit(X, y)

    buffer = io.StringIO()
    buffer.write(f"Etiquetas del modelo: {list(model.classes_)}\n")
    buffer.write(f"Características: {', '.join(FEATURE_COLUMNS)}\n")
    buffer.write("-----------------------------------------\n")
    buffer.write(sk_tree.export_text(model, feature_names=FEATURE_COLUMNS))
    rules = buffer.getvalue()

    return model, rules, metrics


def annotate_criticality(indicators: dict) -> pd.DataFrame:
    """Tabla de resumen de la criticidad de cada nodo en el escenario."""
    features = build_features(indicators)
    if features.empty:
        return features
    return features[["node", "degree", "affected_edges", "blocked_edges", "isolated", "criticality"]]
