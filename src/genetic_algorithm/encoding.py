"""
Codificação dos hiperparâmetros como genes para o Algoritmo Genético.

Cada "indivíduo" é um dicionário de hiperparâmetros. O espaço de busca é
definido por tipo (int, float, categorical) e limites para cada gene.

Modelos suportados: random_forest, logistic_regression, svm.

Uso:
    from src.genetic_algorithm.encoding import SEARCH_SPACES, random_individual

    individuo = random_individual("random_forest")
    # {'n_estimators': 142, 'max_depth': 12, ...}
"""

from __future__ import annotations

import random
from typing import Any

# ---------------------------------------------------------------------------
# Espaços de busca por modelo
# ---------------------------------------------------------------------------
# Cada gene é definido por um dict com:
#   type: "int" | "float" | "categorical"
#   low, high: limites para int/float
#   choices: lista de opções para categorical

SEARCH_SPACES: dict[str, dict[str, dict]] = {
    "random_forest": {
        "n_estimators":      {"type": "int",         "low": 10,   "high": 500},
        "max_depth":         {"type": "categorical",  "choices": [None, 5, 10, 15, 20, 30]},
        "min_samples_split": {"type": "int",          "low": 2,    "high": 20},
        "min_samples_leaf":  {"type": "int",          "low": 1,    "high": 10},
        "max_features":      {"type": "categorical",  "choices": ["sqrt", "log2", None]},
    },
    "logistic_regression": {
        "C":        {"type": "float",       "low": 0.001, "high": 100.0},
        "solver":   {"type": "categorical", "choices": ["lbfgs", "liblinear", "saga"]},
        "max_iter": {"type": "int",         "low": 100,   "high": 2000},
    },
    "svm": {
        "C":      {"type": "float",       "low": 0.01, "high": 100.0},
        "kernel": {"type": "categorical", "choices": ["rbf", "poly", "sigmoid"]},
        "gamma":  {"type": "categorical", "choices": ["scale", "auto"]},
    },
}


def _sample_gene(gene_def: dict) -> Any:
    """Amostra um valor aleatório para um gene dado sua definição."""
    if gene_def["type"] == "int":
        return random.randint(gene_def["low"], gene_def["high"])
    elif gene_def["type"] == "float":
        return random.uniform(gene_def["low"], gene_def["high"])
    elif gene_def["type"] == "categorical":
        return random.choice(gene_def["choices"])
    else:
        raise ValueError(f"Tipo de gene desconhecido: {gene_def['type']}")


def random_individual(model_name: str) -> dict[str, Any]:
    """Gera um indivíduo (hiperparâmetros) aleatório para o modelo dado."""
    space = SEARCH_SPACES[model_name]
    return {gene: _sample_gene(defn) for gene, defn in space.items()}


def random_population(model_name: str, size: int) -> list[dict[str, Any]]:
    """Gera uma população de `size` indivíduos aleatórios."""
    return [random_individual(model_name) for _ in range(size)]


def decode_to_sklearn_params(model_name: str, individual: dict[str, Any]) -> dict[str, Any]:
    """
    Converte um indivíduo para o formato esperado pelo sklearn (prefixo 'model__').

    Uso:
        params = decode_to_sklearn_params("random_forest", individuo)
        pipeline.set_params(**params)
    """
    return {f"model__{k}": v for k, v in individual.items()}
