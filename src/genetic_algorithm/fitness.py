"""
Função fitness para o Algoritmo Genético.

Avalia um indivíduo (conjunto de hiperparâmetros) treinando o modelo
com validação cruzada estratificada e retornando o F1-macro médio.

Métrica principal: F1-macro — adequada para problema multiclasse
com desbalanceamento moderado (low 44% / mid 31% / high 25%).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.genetic_algorithm.encoding import decode_to_sklearn_params

CV_FOLDS = 5
RANDOM_STATE = 42


def evaluate_fitness(
    individual: dict[str, Any],
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scoring: str = "f1_macro",
    n_splits: int = CV_FOLDS,
) -> float:
    """
    Avalia o fitness de um indivíduo via validação cruzada estratificada.

    Parâmetros
    ----------
    individual : dict de hiperparâmetros (genes decodificados)
    model_name : nome do modelo ("random_forest", "logistic_regression", "svm")
    pipeline   : pipeline sklearn base (sem hiperparâmetros ajustados)
    X_train    : features de treino
    y_train    : alvo de treino (int64)
    scoring    : métrica de avaliação (padrão: f1_macro)
    n_splits   : número de folds da CV estratificada

    Retorno
    -------
    float — média do F1-macro nos k folds (entre 0 e 1)
    """
    params = decode_to_sklearn_params(model_name, individual)

    try:
        pipeline.set_params(**params)
    except ValueError as e:
        # Parâmetro inválido para este estimador — retorna fitness mínimo
        return 0.0

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    try:
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )
        return float(np.mean(scores))
    except Exception:
        return 0.0


def evaluate_population(
    population: list[dict[str, Any]],
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> list[float]:
    """
    Avalia todos os indivíduos de uma população.

    Retorna lista de fitness na mesma ordem da população.
    """
    return [
        evaluate_fitness(ind, model_name, pipeline, X_train, y_train)
        for ind in population
    ]
