"""
Loop principal do Algoritmo Genético.

Estratégia:
- Inicialização aleatória da população
- Elitismo: o melhor indivíduo de cada geração é preservado automaticamente
- Por geração: seleção por torneio → cruzamento uniforme → mutação
- Critério de parada: número fixo de gerações
- Histórico completo para análise de convergência

Uso:
    from src.genetic_algorithm.ga import run_genetic_algorithm

    resultado = run_genetic_algorithm(
        model_name="random_forest",
        pipeline=pipeline,
        X_train=X_train,
        y_train=y_train,
        population_size=30,
        generations=20,
        mutation_rate=0.1,
        crossover_rate=0.8,
    )
    print(resultado["best_individual"])
    print(resultado["best_fitness"])
"""

from __future__ import annotations

import random
import time
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src.genetic_algorithm.encoding import random_population
from src.genetic_algorithm.fitness import evaluate_population
from src.genetic_algorithm.operators import tournament_selection, uniform_crossover, mutate


def run_genetic_algorithm(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    population_size: int = 30,
    generations: int = 20,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.8,
    tournament_size: int = 3,
    elitism: bool = True,
    random_state: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Executa o Algoritmo Genético para otimização de hiperparâmetros.

    Parâmetros
    ----------
    model_name      : nome do modelo a otimizar ("random_forest", etc.)
    pipeline        : pipeline sklearn base (sem hiperparâmetros ajustados)
    X_train         : features de treino
    y_train         : alvo de treino (int64)
    population_size : número de indivíduos por geração
    generations     : número de gerações
    mutation_rate   : probabilidade de mutação por gene (0 a 1)
    crossover_rate  : probabilidade de cruzamento entre dois pais (0 a 1)
    tournament_size : número de candidatos no torneio de seleção
    elitism         : se True, preserva o melhor indivíduo de cada geração
    verbose         : se True, imprime progresso por geração

    Retorno
    -------
    dict com:
        best_individual : dict de hiperparâmetros do melhor indivíduo encontrado
        best_fitness    : float — melhor F1-macro obtido
        history         : list de dicts com métricas por geração
        config          : configuração do experimento
        elapsed_seconds : tempo total de execução
    """
    random.seed(random_state)
    start = time.time()

    # --- Inicialização ---
    population = random_population(model_name, population_size)
    history = []
    best_individual = None
    best_fitness = -1.0

    if verbose:
        print(f"\n{'='*55}")
        print(f"Algoritmo Genético — {model_name}")
        print(f"Pop: {population_size} | Gerações: {generations} | "
              f"Mutação: {mutation_rate} | Cruzamento: {crossover_rate}")
        print(f"{'='*55}")

    for gen in range(generations):
        # --- Avaliação ---
        fitness_scores = evaluate_population(population, model_name, pipeline, X_train, y_train)

        # --- Métricas da geração ---
        gen_best_idx = max(range(population_size), key=lambda i: fitness_scores[i])
        gen_best_fitness = fitness_scores[gen_best_idx]
        gen_mean_fitness = sum(fitness_scores) / population_size

        if gen_best_fitness > best_fitness:
            best_fitness = gen_best_fitness
            best_individual = population[gen_best_idx].copy()

        history.append({
            "generation":    gen + 1,
            "best_fitness":  round(gen_best_fitness, 4),
            "mean_fitness":  round(gen_mean_fitness, 4),
            "global_best":   round(best_fitness, 4),
        })

        if verbose:
            print(f"[Gen {gen+1:>3}/{generations}] "
                  f"best={gen_best_fitness:.4f} | "
                  f"mean={gen_mean_fitness:.4f} | "
                  f"global_best={best_fitness:.4f}")

        # Última geração: não cria nova população
        if gen == generations - 1:
            break

        # --- Nova geração ---
        new_population = []

        # Elitismo: preserva o melhor
        if elitism:
            new_population.append(best_individual.copy())

        # Preenche o resto com filhos
        while len(new_population) < population_size:
            parent_a = tournament_selection(population, fitness_scores, tournament_size)
            parent_b = tournament_selection(population, fitness_scores, tournament_size)
            child_a, child_b = uniform_crossover(parent_a, parent_b, crossover_rate)
            child_a = mutate(child_a, model_name, mutation_rate)
            child_b = mutate(child_b, model_name, mutation_rate)
            new_population.append(child_a)
            if len(new_population) < population_size:
                new_population.append(child_b)

        population = new_population

    elapsed = round(time.time() - start, 1)

    if verbose:
        print(f"\nMelhor indivíduo: {best_individual}")
        print(f"Melhor F1-macro:  {best_fitness:.4f}")
        print(f"Tempo total:      {elapsed}s")

    return {
        "best_individual": best_individual,
        "best_fitness":    best_fitness,
        "history":         history,
        "config": {
            "model_name":       model_name,
            "population_size":  population_size,
            "generations":      generations,
            "mutation_rate":    mutation_rate,
            "crossover_rate":   crossover_rate,
            "tournament_size":  tournament_size,
            "elitism":          elitism,
            "random_state":     random_state,
        },
        "elapsed_seconds": elapsed,
    }
