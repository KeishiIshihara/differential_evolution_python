
# =====================================================
#  Differential Evolution implementation in Python
# 
#  Reference: 
#  https://en.wikipedia.org/wiki/Differential_evolution
#
#  Auther: Keishi Ishihara
# =====================================================

from __future__ import print_function
import random, copy
from decimal import Decimal
from operator import attrgetter
import numpy as np
import math
import matplotlib.pyplot as plt
import csv
import os


# main function
def main():

    # Objective function 
    # Note: it might possibly take time to call if difine here.
    def Schwefel_function(params):
        """Objective function given by the exercise."""
        A = 0
        for p in params:
            rad = math.sqrt(abs(p))
            A += p * math.sin(rad)
        return Decimal('418.9829') * Decimal(str(len(params)))  -  Decimal(str(A))


    best_agent = DifferentialEvolution(pop_size=300, 
                                       num_gen=300, 
                                       x_range=[-500,500], 
                                       F=1.0, 
                                       CR=0.5, 
                                       dim=5,
                                       obj_f=Schwefel_function,
                                       print_detail=True)

    print('Ans. ',best_agent.getFitness())


# DE algorithm here
def DifferentialEvolution(pop_size=100, num_gen=50, x_range=None, F=1, CR=0.5, dim=2, obj_f=None, print_detail=False):
    """Differential Evolution.
    Basically, this function will try to minimize the Schwefel's function.
    This code is based on the description of wikipedia page.

    For those arguments, see below.
    """
    ## For better understanding, redifine the params here
    POPULATION_SIZE = pop_size   # number of agents in a population.
    NUM_GEN         = num_gen    # number of generation loops.
    X_RANGE         = x_range    # domain limit.
    F               = F          # a coefficience called differential weight.
    CR              = CR         # crossover probability.
    N               = dim        # dimention of search-space.
    OBJ_F           = obj_f      # objective function.


    os.makedirs('results', exist_ok=True)
    result_arr = [] # for storing the results of each generation.
    random.seed(64)

    # --- step 1 : Initialize all Agent x randomly in the search-space.
    population = create_population(POPULATION_SIZE, N, X_RANGE, OBJ_F)
    best = min(population, key=attrgetter('fitness'))

    # --- Until the termination criterion met, repeat following.
    if print_detail:
        print('Generation loop starts. ')
        print("Generation: 0. Initial best fitness: {}".format(best.getFitness()))

    g_ = 0
    # --- Now it loops until adequate fitness was satisfied.
    while best.getFitness() > 1:
        g_ += 1
        # --- For each Agent x in the population:
        for x in range(POPULATION_SIZE):

            # --- step 2 : Pick up there different Agents' index from the population randomly.
            #              As well as those must be distinct from agent x.
            a, b, c = np.random.choice(np.delete(range(0, POPULATION_SIZE),x), 3, replace=False)

            R = random.randint(0, N-1) # Pick an random index R from 0 to N-1.

            # --- step 3 : Create new candidate Agent using those Agent a, b and c.
            candidate = []
            # Each params of new Agent will be 
            for i in range(N):
                if (random.random() < CR) or (i == R):
                    # calculate the differential vector.
                    y = population[a].param[i] + F * (population[b].param[i] - population[c].param[i])
                    # if the candidate is in out of range, make it within the range.
                    y = X_RANGE[0] if y < X_RANGE[0] else X_RANGE[1] if y > X_RANGE[1] else y
                    candidate.append(y)
                else:
                    candidate.append(population[x].param[i])
            
            # --- step 4 : Update Agent x with candidate if the fitness of candidate is better than x's
            if fitness(candidate, OBJ_F) < fitness(population[x].getParam(), OBJ_F):
                population[x].setParam(candidate)
                population[x].setFitness(fitness(candidate, OBJ_F))

        # Pick up best and worst agent from current population for ploting.
        best  = min(population, key=attrgetter('fitness'))
        worst = max(population, key=attrgetter('fitness'))

        result_arr.append([best.getFitness(), worst.getFitness()])

        if print_detail:
            print('Generation: {}. Best fitness {}'.format(g_, best.getFitness()))
        
        if g_ % 50 == 0:
            plot_curve(np.array(result_arr), pop_size=POPULATION_SIZE, f=F, cr=CR, d=dim)


    # --- Plot results.
    plot_curve(np.array(result_arr), pop_size=POPULATION_SIZE, f=F, cr=CR, d=dim)

    # --- Print final result.
    best_agent = min(population, key=attrgetter('fitness'))
    print('Params of best agent: {}'.format(best_agent.getParam()))
    print('The best fitness:     {}\n--'.format(best_agent.getFitness()))

    # --- Summarize result to csv.
    with open('results/summary_dim{}.csv'.format(N),'w') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        writer.writerow(['population size', POPULATION_SIZE])
        writer.writerow(['num generation', NUM_GEN])
        writer.writerow(['F', F])
        writer.writerow(['CR', CR])
        writer.writerow(['N', N])
        writer.writerow([' '])
        writer.writerow(['final param'] + best_agent.getParam())
        writer.writerow(['final score', best_agent.getFitness()])

    return best_agent


class Agent(object):
    """Agent class.

    # Params
        param: holds parameters of Agent.
        fitness: holds the fitness value of this Agent.
    """

    param = None
    fitness = None

    def __init__(self, param, fitness):
        self.param = param
        self.fitness = fitness

    def __len__(self):
        return len(self.param)

    def getParam(self):
        return self.param

    def getFitness(self):
        return self.fitness

    def setParam(self, param):
        self.param = param

    def setFitness(self, fitness):
        self.fitness = fitness


def create_agent(length, _range, obj_f):
    """This creates binary encoded parameters list (agent).

    # Argument
        length: the length of parameters.
        _range: the range of parameter.
        obj_f: objective function.
    # Return
        Agent class
    """
    param = [random.uniform(_range[0], _range[1]) for i in range(length)] # param
    return Agent(param, fitness(param, obj_f))

def create_population(population_size, num_params, _range, obj_f):
    """This creates a population of agents.

    # Argument
        population_size: the number of agents in population.
        num_params: the number of params.
        _range: the range of parameter.
        obj_f: objective function.
    # Return
        list, population of agents.
    """
    pop = [create_agent(num_params, _range, obj_f) for i in range(population_size)]
    return pop

def fitness(params, obj_func):
    """Calulates fitness on each agents in a population.

    # Argument
        params: parameters in Agent class.
        obj_func: objective function.
    # Return
        float, value of fitness.
    """
    return obj_func(params)

def plot_curve(results, pop_size=None, f=None, cr=None, d=None):
    if len(results) > 1:
        # To make sure that there are more than 2 generations.
        X = np.arange(1, len(results)+1)

        _, ax  = plt.subplots(ncols=1)
        ax.plot(X, results[:,0] ,label='best fitness')
        ax.plot(X, results[:,1] ,label='worst fitness')
        ax.set_title('[DE] Schwefel\'s function (POP={},F={},CR={},dim={})'.format(pop_size,f,cr,d))
        ax.set_xlabel('Generation')
        ax.set_ylabel('Fitness')
        ax.grid()
        ax.legend(loc='best')
        plt.savefig('results/fitness.png')
        plt.close()


if __name__ == '__main__':
    main()
