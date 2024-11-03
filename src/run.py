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
        self.env = CarRacing(render_mode=render_mode, continuous=continuous, lap_complete_percent=lap_complete_percent, verbose=True)
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
    parser = argparse.ArgumentParser(prog="CarRacing", description="CarRacing AI learning")
    parser.add_argument("-d", "--do", type=str,choices=["train", "evaluate"], help="provide action", default="train")
    parser.add_argument("-a","--algorithm", type=str,choices=["GA", "GP"], help="provide algorithm", default="GA")
    parser.add_argument("-l","--load", type=str, help="name of document obtaining individual", default="src\\best_individual_fitness.txt")
    parser.add_argument("-g", "--generations", type=int, help="number of generations", default=20)
    parser.add_argument("-p", "--populations", type=int, help="number of populations", default=20)
    parser.add_argument("-st", "--steps", type=int, help="number of steps if 0 continuous regime", default=1000) # TODO add continuous regime for GA
    parser.add_argument("-m", "--mutation", type=float, help="probability of mutation", default=0.2)
    parser.add_argument("-c", "--continuous", type=bool, help="bool to choose between discrete and box action space", default=True)
    parser.add_argument("-co", "--crossover", type=float, help="probability of crossover", default=0.5)
    parser.add_argument("-t", "--tournament", type=int, help="size of tournament pool", default=5)
    parser.add_argument("-s", "--save", type=str, help="name of document obtaining individuals",default="src\\best_individual_fitness.txt", required=False)
    parser.add_argument("-gl", "--genome_length", type=int, help="genome length", default=500)
    parser.add_argument("--show", type=bool, help="show best individual", default=False)
    args = parser.parse_args()

    continuous = args.continuous
    env = Environment(continuous=continuous, render_mode="human")

    if args.do == "train":
        if args.algorithm == "GA":
            ga = GeneticAlgorithm(env,population_size=args.populations, genome_length=args.genome_length, generations=args.generations, mutation_rate=args.mutation,
                                  crossover_rate=args.crossover, tournament_size=args.tournament, continuous=continuous)
            best_individual = ga.run()
            if args.show:
                render_env = Environment(continuous=continuous)
                render_ga = GeneticAlgorithm(render_env, continuous=continuous)
                fitness = render_ga.evaluate_best(best_individual)
            else:
                fitness = ga.evaluate_best(best_individual)
            if args.save:
                save_best_individual(best_individual, fitness=fitness, file_name=args.save)
        if args.algorithm == "GP":
            gp = GeneticProgramming(env, population_size=args.populations, generations=args.generations, mutation_rate=args.mutation,crossover_rate=args.crossover,
                                    tournament_size=args.tournament, continuous=continuous)
            best_individual = gp.run()
            print(best_individual)



    if args.do == "evaluate":
        file_load = args.load
        best_individual, total_reward = load_best_individual(file_load)
        print(f"{total_reward} : {best_individual}")
        render_env = Environment(continuous=continuous, render_mode="human")
        if args.algorithm == "GA":
            render_ga = GeneticAlgorithm(render_env, continuous=continuous)
            fit = render_ga.evaluate_best(best_individual)
        else:
            exit()