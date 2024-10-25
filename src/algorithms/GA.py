import numpy as np
import random

# TODO Pokud fitness spadne pod -10 ukončit run hodit tot reward na -100 a done = False



class GeneticAlgorithm:
    def __init__(self, env, population_size=50, genome_length=1000, generations=10, mutation_rate=0.1, crossover_rate=0.7, continuous=True):
        self.env = env
        self.population_size = population_size
        self.genome_length = genome_length
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.continuous = continuous

        self.population = [self.random_genome() for _ in range(self.population_size)]

    def random_genome(self):
        if not self.continuous:
            return np.random.choice(self.env.action_space.n, self.genome_length)
        else:
            return np.random.choice(len(self.env.action_space), self.genome_length)

    def fitness(self, genome):
        obs = self.env.reset()
        total_reward = 0
        done = False

        for action in genome:
            obs,reward, done, truncated, _ = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break

        return total_reward

    def tournament_selection(self, fitnesses):
        tournament_size = 10
        selected = random.sample(range(len(self.population)), tournament_size)
        best_individual = max(selected, key=lambda idx: fitnesses[idx])
        return self.population[best_individual]

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            cross_point = random.randint(1, self.genome_length - 1)
            child = np.concatenate((parent1[:cross_point], parent2[cross_point:]), axis=0)
        else:
            child = parent1.copy()
        return child

    def mutate(self, genome):
        for i in range(len(genome)):
            if random.random() < self.mutation_rate:
                if not self.continuous:
                    genome[i] = random.choice(range(self.env.action_space.n))
                else:
                    genome[i] = random.choice(range(len(self.env.action_space)))
        return genome


    def run(self):
        for generation in range(self.generations):
            # Evaluate the fitness of the population
            fitnesses = []
            for individual in self.population:
                fitnesses.append(self.fitness(individual))

            # Keep track of the best individual in the current generation
            best_fitness = max(fitnesses)
            best_individual = self.population[fitnesses.index(best_fitness)]
            print(f"Generation {generation}, Best Fitness: {best_fitness}")

            # Create the next generation
            new_population = []
            while len(new_population) < self.population_size:
                # Select two parents
                parent1 = self.tournament_selection(fitnesses)
                parent2 = self.tournament_selection(fitnesses)

                # Apply crossover and mutation to create two children
                child1 = self.crossover(parent1, parent2)
                child2 = self.crossover(parent2, parent1)

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.append(child1)
                new_population.append(child2)

            # Replace the population with the new generation
            self.population = new_population[:self.population_size]

        # Return the best individual after the final generation
        return best_individual

    def evaluate_best(self, best_individual):
        obs = self.env.reset()
        total_reward = 0
        done = False
        for action in best_individual:
            obs, reward, done, truncated, _ = self.env.step(action)
            total_reward += reward
            self.env.render()
            if done or truncated:
                break
        print(f"Best Fitness: {total_reward}")
        return total_reward