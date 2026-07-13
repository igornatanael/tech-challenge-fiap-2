"""
Testes para src/genetic_algorithm/encoding.py

Cobre: SEARCH_SPACES, random_individual, random_population, decode_to_sklearn_params.
"""

import random

import pytest

from src.genetic_algorithm.encoding import (
    SEARCH_SPACES,
    decode_to_sklearn_params,
    random_individual,
    random_population,
)

MODEL = "random_forest"
SPACE = SEARCH_SPACES[MODEL]


# ---------------------------------------------------------------------------
# random_individual
# ---------------------------------------------------------------------------


def test_random_individual_keys():
    """Indivíduo tem exatamente as keys do SEARCH_SPACES["random_forest"]."""
    random.seed(42)
    individual = random_individual(MODEL)
    assert set(individual.keys()) == set(SPACE.keys())


def test_random_individual_bounds_int():
    """Genes do tipo int respeitam low/high."""
    random.seed(42)
    for _ in range(20):
        individual = random_individual(MODEL)
        for gene, defn in SPACE.items():
            if defn["type"] == "int":
                assert defn["low"] <= individual[gene] <= defn["high"], (
                    f"Gene '{gene}' = {individual[gene]} fora de [{defn['low']}, {defn['high']}]"
                )


def test_random_individual_bounds_float():
    """Genes do tipo float respeitam low/high."""
    space = SEARCH_SPACES["logistic_regression"]
    random.seed(42)
    for _ in range(20):
        individual = random_individual("logistic_regression")
        for gene, defn in space.items():
            if defn["type"] == "float":
                assert defn["low"] <= individual[gene] <= defn["high"], (
                    f"Gene '{gene}' = {individual[gene]} fora de [{defn['low']}, {defn['high']}]"
                )


def test_random_individual_categorical():
    """Genes do tipo categorical estão nas choices definidas."""
    random.seed(42)
    for _ in range(20):
        individual = random_individual(MODEL)
        for gene, defn in SPACE.items():
            if defn["type"] == "categorical":
                assert individual[gene] in defn["choices"], (
                    f"Gene '{gene}' = {individual[gene]!r} não está em {defn['choices']}"
                )


# ---------------------------------------------------------------------------
# random_population
# ---------------------------------------------------------------------------


def test_random_population_size():
    """random_population retorna lista do tamanho correto."""
    random.seed(42)
    pop = random_population(MODEL, size=10)
    assert isinstance(pop, list)
    assert len(pop) == 10


# ---------------------------------------------------------------------------
# decode_to_sklearn_params
# ---------------------------------------------------------------------------


def test_decode_to_sklearn_params_keys():
    """decode_to_sklearn_params retorna params com prefixo 'model__'."""
    random.seed(42)
    individual = random_individual(MODEL)
    params = decode_to_sklearn_params(MODEL, individual)

    assert set(params.keys()) == {f"model__{k}" for k in individual.keys()}
    # Valores devem ser os mesmos do indivíduo original
    for gene, val in individual.items():
        assert params[f"model__{gene}"] == val
