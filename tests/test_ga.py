"""
Testes para src/genetic_algorithm/ga.py

Cobre: run_genetic_algorithm — estrutura do resultado, histórico, monotonicidade e
reprodutibilidade. Usa dataset sintético mínimo para manter a execução rápida.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.genetic_algorithm.ga import run_genetic_algorithm

# ---------------------------------------------------------------------------
# Dataset sintético mínimo (50 amostras, 6 features, 3 classes)
# ---------------------------------------------------------------------------

np.random.seed(42)
X_TRAIN = pd.DataFrame(
    np.random.rand(50, 6),
    columns=["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"],
)
Y_TRAIN = pd.Series(np.random.randint(0, 3, 50), dtype="int64")

# ---------------------------------------------------------------------------
# Pipeline mínimo reutilizável
# ---------------------------------------------------------------------------

GA_KWARGS = dict(
    model_name="random_forest",
    X_train=X_TRAIN,
    y_train=Y_TRAIN,
    population_size=5,
    generations=3,
    mutation_rate=0.1,
    crossover_rate=0.8,
    tournament_size=3,
    verbose=False,
    random_state=42,
)


def _make_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42, n_jobs=1)),
    ])


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


def test_run_genetic_algorithm_returns_keys():
    """Resultado contém as keys esperadas."""
    result = run_genetic_algorithm(pipeline=_make_pipeline(), **GA_KWARGS)

    expected_keys = {"best_individual", "best_fitness", "history", "config", "elapsed_seconds"}
    assert expected_keys.issubset(result.keys())


def test_run_genetic_algorithm_history_length():
    """len(history) é igual ao número de gerações."""
    result = run_genetic_algorithm(pipeline=_make_pipeline(), **GA_KWARGS)

    assert len(result["history"]) == GA_KWARGS["generations"]


def test_run_genetic_algorithm_best_fitness_improves_or_equal():
    """global_best nunca diminui entre gerações consecutivas."""
    result = run_genetic_algorithm(pipeline=_make_pipeline(), **GA_KWARGS)

    history = result["history"]
    for i in range(1, len(history)):
        assert history[i]["global_best"] >= history[i - 1]["global_best"], (
            f"global_best diminuiu da geração {i} ({history[i-1]['global_best']}) "
            f"para {i+1} ({history[i]['global_best']})"
        )


def test_run_genetic_algorithm_reproducible():
    """Dois runs com o mesmo random_state produzem o mesmo best_individual."""
    result_a = run_genetic_algorithm(pipeline=_make_pipeline(), **GA_KWARGS)
    result_b = run_genetic_algorithm(pipeline=_make_pipeline(), **GA_KWARGS)

    assert result_a["best_individual"] == result_b["best_individual"]
