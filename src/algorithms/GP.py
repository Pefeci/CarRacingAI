from deap import base, creator, tools, gp, algorithms
import operator
import numpy as np
import random
from gymnasium.envs.box2d import car_dynamics
from algorithms.CarRacingFeatureExtractor import CarRacingFeatureExtractor

# TODO bylo by fajn dodelat evaluate best pro zobrazeni nejlepsiho individua
def protected_div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


class GeneticProgramming:
    def __init__(self, env, population_size=50, generations=100, tournament_size=3, mutation_rate=0.2, crossover_rate=0.5,max_steps=1000, continuous=True):
        self.env = env
        self.max_steps = max_steps
        self.continuous = continuous
        self.crossover_rate = crossover_rate
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.toolbox = base.Toolbox()

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


        self.pset = gp.PrimitiveSet("MAIN", 3)  # Three inputs: speed, angle, reward
        self.pset.addPrimitive(operator.add, 2)
        self.pset.addPrimitive(operator.sub, 2)
        self.pset.addPrimitive(operator.mul, 2)
        self.pset.addPrimitive(protected_div, 2)
        self.pset.addPrimitive(np.sin, 1)
        self.pset.addPrimitive(np.cos, 1)

        self.pset.renameArguments(ARG0="speed")
        self.pset.renameArguments(ARG1="angle")
        self.pset.renameArguments(ARG2="last_reward")

        self.toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=1, max_=2)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=self.pset)

        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr, pset=self.pset)

    def evaluate(self, individual):
        func = self.toolbox.compile(expr=individual)

        obs = self.env.reset()
        done = False
        total_reward = 0
        crfe = CarRacingFeatureExtractor()
        grass_counter = 0
        last_reward = 0

        while total_reward >= 0:
            speed, angle, _ = crfe.extract_features(obs)
            action_value = func(speed, angle, last_reward)
            action = self.map_function(action_value)

            obs, reward, done, truncated, _ = self.env.step(action)
            last_reward = reward
            last_total_reward = total_reward
            total_reward += reward

            if last_total_reward > total_reward:
                grass_counter += 1
                if grass_counter >= 330:
                    break
            else:
                grass_counter = 0

            if done:
                break
        return total_reward,

    def map_function(self, action_value):
        if not self.continuous:
            if action_value < -0.5: #steer left
                return 1
            elif action_value < 0: #break
                return 4
            elif action_value < 0.5: #gas
                return 3
            else:       #steer right
                return 2
        else:
            if action_value < -0.5: #steer left
                #return np.array([-1,0,0]).astype(np.float32)
                return 1
            elif action_value < 0: #break
                #return np.array([0,0,1]).astype(np.float32)
                return 4
            elif action_value < 0.5: #gas
                #return np.array([0,1,0]).astype(np.float32)
                return 3
            else:       #steer right
                #return np.array([1,0,0]).astype(np.float32)
                 return 2




    def run(self):
        population = self.toolbox.population(n=self.population_size)

        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        for gen in range(self.generations):

            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.crossover_rate:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutation_rate:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate new individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Replace population
            population[:] = offspring

            # Gather stats
            fits = [ind.fitness.values[0] for ind in population]
            best_ind = tools.selBest(population, 1)[0]
            print(f"Generation {gen} - Best Fitness: {best_ind.fitness.values[0]}: {best_ind}")

            # Return best individual
        return tools.selBest(population, 1)[0]

