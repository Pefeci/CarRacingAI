import gymnasium as gym
import argparse
from gymnasium import spaces
import numpy as np
import re


class Environment:
    def __init__(self, lap_complete_percent=0.95, render_mode="rgb_array", continuous=False, algorithm="GA"):
        self.continuous = continuous
        self.env = gym.make('CarRacing-v3', render_mode=render_mode, continuous=continuous, lap_complete_percent=lap_complete_percent)

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

def save_best_individual(best_individual, fitness):
    with open("best_individual_fitness.txt", "a") as f:
        f.write(str(best_individual) + "\t"
                + str(fitness) + "\t\n")

def load_best_individual(file_name):
    with open("best_individual_fitness.txt", "r") as f:
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
                is_fitness = False
        best_fitness = max(fitnesses)
        return individuals[best_fitness], best_fitness





if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="CarRacing", description="CarRacing AI learning")
    parser.add_argument("-d", "--do", type=str,choices=["train", "evaluate"], help="provide action: {train, evaluate}", default="train")
    parser.add_argument("-a","--algorithm", type=str,choices=["GA"], help="provide algorithm: {GA}", default="GA")
    parser.add_argument("-l","--load", type=str, help="name of document obtaining individual", default="best_individual_fitness.txt")
    parser.add_argument("-g", "--generations", type=int, help="number of generations", default=10)
    parser.add_argument("-p", "--populations", type=int, help="number of populations", default=10)
    parser.add_argument("-s", "--steps", type=int, help="number of steps if 0 continuous regime", default=1000) # TODO add continuous regime for GA
    parser.add_argument("-m", "--mutation", type=float, help="probability of mutation", default=0.2)
    parser.add_argument("-c", "--continuous", type=bool, help="bool to choose between discrete and box action space")
    parser.add_argument("-co", "--crossover", type=float, help="probability of crossover", default=0.5)
    parser.add_argument("-t", "--tournament", type=int, help="size of tournament pool", default=5)
    args = parser.parse_args()





    best_individual, total_reward = load_best_individual("best_individual_fitness.txt")
    print(f"{total_reward} : {best_individual}")


    # continuous = True
    # # parser = argparse.ArgumentParser()
    # # env = Environment(continuous=continuous)
    # # ga = GeneticAlgorithm(env, continuous=continuous)
    # #
    # render_env=Environment(continuous=continuous, render_mode="human")
    # render_ga = GeneticAlgorithm(render_env, continuous=continuous)
    # #
    # # best_individual = ga.run()
    # #
    # # print("_________________TESTING BEST INDIVIDUAL___________________")
    # fit = render_ga.evaluate_best(best_individual)
    # # save_best_individual(best_individual, fit)