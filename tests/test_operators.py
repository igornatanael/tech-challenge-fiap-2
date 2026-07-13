"""
Testes para src/genetic_algorithm/operators.py

Cobre: tournament_selection, uniform_crossover, mutate.
"""

import random

import pytest

from src.genetic_algorithm.encoding import SEARCH_SPACES, random_population
from src.genetic_algorithm.operators import mutate, tournament_selection, uniform_crossover

MODEL = "random_forest"
SPACE = SEARCH_SPACES[MODEL]


def _make_population(size: int = 6, seed: int = 42) -> list[dict]:
    random.seed(seed)
    return random_population(MODEL, size)


# ---------------------------------------------------------------------------
# tournament_selection
# ---------------------------------------------------------------------------


def test_tournament_selection_returns_individual():
    """tournament_selection retorna dict com as mesmas keys da população."""
    population = _make_population()
    fitness_scores = [0.5, 0.6, 0.7, 0.4, 0.8, 0.3]
    random.seed(42)
    winner = tournament_selection(population, fitness_scores, tournament_size=3)

    assert isinstance(winner, dict)
    assert set(winner.keys()) == set(SPACE.keys())


def test_tournament_selection_winner_is_best():
    """Com tournament_size igual ao tamanho da população, retorna o melhor indivíduo."""
    population = _make_population(size=6)
    fitness_scores = [0.1, 0.9, 0.3, 0.4, 0.2, 0.5]
    best_idx = fitness_scores.index(max(fitness_scores))  # índice 1

    random.seed(42)
    winner = tournament_selection(population, fitness_scores, tournament_size=6)

    assert winner == population[best_idx]


# ---------------------------------------------------------------------------
# uniform_crossover
# ---------------------------------------------------------------------------


def test_uniform_crossover_no_crossover():
    """Com crossover_rate=0.0, uniform_crossover retorna cópias dos pais."""
    population = _make_population(size=2)
    parent_a, parent_b = population[0], population[1]

    # crossover_rate=0.0 faz random.random() > 0.0 ser sempre True
    # → retorna cópias sem cruzamento
    random.seed(42)
    child_a, child_b = uniform_crossover(parent_a, parent_b, crossover_rate=0.0)

    assert child_a == parent_a
    assert child_b == parent_b


def test_uniform_crossover_keys_preserved():
    """Os filhos têm as mesmas keys dos pais após o cruzamento."""
    population = _make_population(size=2)
    parent_a, parent_b = population[0], population[1]

    random.seed(42)
    child_a, child_b = uniform_crossover(parent_a, parent_b, crossover_rate=1.0)

    assert set(child_a.keys()) == set(SPACE.keys())
    assert set(child_b.keys()) == set(SPACE.keys())


# ---------------------------------------------------------------------------
# mutate
# ---------------------------------------------------------------------------


def test_mutate_keys_preserved():
    """O mutante tem as mesmas keys do indivíduo original."""
    random.seed(42)
    population = _make_population(size=1)
    individual = population[0]

    mutant = mutate(individual, MODEL, mutation_rate=0.5)

    assert set(mutant.keys()) == set(SPACE.keys())


def test_mutate_rate_zero_unchanged():
    """Com mutation_rate=0.0, mutate retorna indivíduo idêntico ao original."""
    random.seed(42)
    population = _make_population(size=1)
    individual = population[0]

    mutant = mutate(individual, MODEL, mutation_rate=0.0)

    assert mutant == individual


def test_mutate_bounds_respected():
    """Com mutation_rate=1.0, genes int/float resultantes respeitam os bounds."""
    random.seed(42)
    population = _make_population(size=1)
    individual = population[0]

    for _ in range(30):
        mutant = mutate(individual, MODEL, mutation_rate=1.0)
        for gene, defn in SPACE.items():
            if defn["type"] == "int":
                assert defn["low"] <= mutant[gene] <= defn["high"], (
                    f"Gene int '{gene}' = {mutant[gene]} fora de [{defn['low']}, {defn['high']}]"
                )
            elif defn["type"] == "float":
                assert defn["low"] <= mutant[gene] <= defn["high"], (
                    f"Gene float '{gene}' = {mutant[gene]} fora de [{defn['low']}, {defn['high']}]"
                )
            elif defn["type"] == "categorical":
                assert mutant[gene] in defn["choices"], (
                    f"Gene categorical '{gene}' = {mutant[gene]!r} não está em {defn['choices']}"
                )
