from deap import base, creator, tools, gp, algorithms
import operator
import numpy as np
import random


def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


class GeneticProgramming:
    def __init__(self, env, population_size=50, generations=100, tournament_size=3, mutation_rate=0.2):
        self.env = env
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.toolbox = base.Toolbox()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


        self.pset = gp.PrimitiveSet("MAIN", 3)  # Three inputs: speed, angle, distance to center
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(protected_div, 2)
        self.pset.addPrimitive(np.sin, 1)
        self.pset.addPrimitive(np.cos, 1)

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr, pset=self.pset)

    def evaluate(self, individual):