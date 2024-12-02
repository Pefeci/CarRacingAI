import gymnasium as gym
import argparse
from gymnasium import spaces
import numpy as np
import re
from gymnasium.envs.box2d import car_dynamics, car_racing, CarRacing
from typing_extensions import Optional

from algorithms.GA import GeneticAlgorithm
from algorithms.GP import GeneticProgramming


class Environment:
    def __init__(self, lap_complete_percent=0.95, render_mode="rgb_array", continuous=False):
        self.continuous = continuous
        #self.env = gym.make('CarRacing-v3', render_mode=render_mode, continuous=continuous, lap_complete_percent=lap_complete_percent)
        self.env = CarRacing(render_mode=render_mode, continuous=continuous, lap_complete_percent=lap_complete_percent, verbose=False)
        self.car: car_dynamics.Car = self.env.car



        if not continuous:
            self.action_space = spaces.discrete.Discrete(5)
        else:
            self.action_space = {
                0: np.array([0, 0, 0]).astype(np.float32),
                1: np.array([-1, 0, 0]).astype(np.float32),
                2: np.array([1, 0, 0]).astype(np.float32),
                3: np.array([0, 1, 0]).astype(np.float32),
                4: np.array([0, 0, 1]).astype(np.float32),
                5: np.array([-1, 0, 1]).astype(np.float32),
                6: np.array([1, 0, 1]).astype(np.float32),
                7: np.array([-1, 1, 0]).astype(np.float32),
                8: np.array([1, 1, 0]).astype(np.float32)
            }

    def reset(self):
        obs, _ = self.env.reset(options={"randomize": False})
        return obs

    def close(self):
        self.env.close()

    def step(self, action):
        if not self.continuous:
            return self.env.step(action)
        else:
            return self.env.step(self.action_space[action])

    def render(self):
        return self.env.render()



def save_best_individual(best_individual, fitness, file_name):
     with open(file_name, "a") as f:
        f.write(str(best_individual) + "\t"
                + str(fitness) + "\t\n")

def load_best_individual(file_name):
    with open(file_name, "r") as f:
        raw_text = f.read()
        individuals = {}
        raw_text = re.split("\n|\t", raw_text)
        print(raw_text)
        individual = []
        fitnesses = []
        is_fitness = False
        is_end = False
        for line in raw_text:
            if not is_fitness:
                if "[" in line:
                    line = line.strip("[")
                if "]" in line:
                    line = line.strip("]")
                    is_end = True
                action_list = line.split(" ")
                if "" in action_list:
                    action_list.remove("")
                for action in action_list:
                    individual.append(int(action))
                if is_end:
                    is_fitness = True
            else:
                fitness = float(line)
                fitnesses.append(fitness)
                is_end = False
                individuals[fitness] = individual
                individual = []
                is_fitness = False
        best_fitness = max(fitnesses)
        return individuals[best_fitness], best_fitness





if __name__ == "__main__":
    continuous = True
    env = Environment(continuous=continuous, render_mode="human")
    gp = GeneticProgramming(env, population_size=10, generations=10, mutation_rate=0.3,crossover_rate=0.2,
                            tournament_size=3, continuous=continuous)
    best_individual = gp.run()
    print(best_individual)
    ren_env = Environment(continuous=continuous, render_mode="human")
    ren_gp = GeneticProgramming(ren_env, population_size=10, generations=10, mutation_rate=0.3, crossover_rate=0.2,
                            tournament_size=3, continuous=continuous)
    ren_gp.evaluate(best_individual)











