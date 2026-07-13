"""
Operadores genéticos: seleção, cruzamento e mutação.

Seleção  — Torneio binário: robusto, simples, sem pressão seletiva excessiva.
Cruzamento — Uniforme: cada gene do filho é herdado de um dos pais com prob. 0.5.
Mutação  — Por gene: int/float sofrem perturbação aleatória; categorical troca de opção.
"""

from __future__ import annotations

import random
from typing import Any

from src.genetic_algorithm.encoding import SEARCH_SPACES, _sample_gene


# ---------------------------------------------------------------------------
# Seleção por torneio
# ---------------------------------------------------------------------------

def tournament_selection(
    population: list[dict[str, Any]],
    fitness_scores: list[float],
    tournament_size: int = 3,
) -> dict[str, Any]:
    """
    Seleciona um indivíduo via torneio binário.

    Amostra `tournament_size` candidatos aleatoriamente e retorna
    o de maior fitness. Favorece bons indivíduos sem eliminar diversidade.
    """
    indices = random.sample(range(len(population)), k=tournament_size)
    winner = max(indices, key=lambda i: fitness_scores[i])
    return population[winner].copy()


# ---------------------------------------------------------------------------
# Cruzamento uniforme
# ---------------------------------------------------------------------------

def uniform_crossover(
    parent_a: dict[str, Any],
    parent_b: dict[str, Any],
    crossover_rate: float = 0.8,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Cruzamento uniforme entre dois pais.

    Se o número aleatório > crossover_rate, retorna cópias dos pais sem cruzar.
    Caso contrário, cada gene do filho é herdado de um dos pais com prob. 0.5.

    Retorna dois filhos.
    """
    if random.random() > crossover_rate:
        return parent_a.copy(), parent_b.copy()

    child_a, child_b = {}, {}
    for gene in parent_a:
        if random.random() < 0.5:
            child_a[gene] = parent_a[gene]
            child_b[gene] = parent_b[gene]
        else:
            child_a[gene] = parent_b[gene]
            child_b[gene] = parent_a[gene]

    return child_a, child_b


# ---------------------------------------------------------------------------
# Mutação por gene
# ---------------------------------------------------------------------------

def mutate(
    individual: dict[str, Any],
    model_name: str,
    mutation_rate: float = 0.1,
) -> dict[str, Any]:
    """
    Aplica mutação gene a gene com probabilidade `mutation_rate`.

    - int: nova amostra aleatória dentro dos bounds
    - float: perturbação gaussiana (std = 20% do range), clipada aos bounds
    - categorical: nova opção aleatória do espaço de busca
    """
    space = SEARCH_SPACES[model_name]
    mutant = individual.copy()

    for gene, defn in space.items():
        if random.random() > mutation_rate:
            continue

        if defn["type"] == "int":
            mutant[gene] = random.randint(defn["low"], defn["high"])

        elif defn["type"] == "float":
            std = (defn["high"] - defn["low"]) * 0.2
            new_val = individual[gene] + random.gauss(0, std)
            mutant[gene] = max(defn["low"], min(defn["high"], new_val))

        elif defn["type"] == "categorical":
            mutant[gene] = random.choice(defn["choices"])

    return mutant
