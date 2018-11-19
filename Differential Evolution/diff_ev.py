import numpy as np
import math
from IPython import embed
from oct2py import octave
import pandas as pd
import matplotlib.pyplot as plt

def differential_evolution(fobj, bounds, use_octave = False, mut=0.9, crossp=0.7, popsize=100, its=100):

    fitness_list = []
    generations = []

    dimensions = len(bounds)
    
    pop = np.random.rand(popsize, dimensions)

    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)

    pop_denorm = min_b + pop * diff

    if(use_octave):
        fitness = np.asarray([octave.feval('peaks', ind) for ind in pop_denorm]) 
    else:
        fitness = np.asarray([octave.feval('rastrigin', ind) for ind in pop_denorm]) 

    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    for i in range(its):

        for j in range(popsize):

            idxs = [idx for idx in range(popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace = False)]

            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp

            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff

            if(use_octave):
                f = octave.feval('peaks', trial_denorm)
            else:
                f = fobj(trial_denorm)

            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial

                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
 
        yield best, fitness[best_idx]
        print("Best individual in generation %s : %f , %f" %  (i+1, best[0], best[1]))
        print("Best individual score in generation %s : %s" %  (i+1, fitness[best_idx]))
        fitness_list.append(fitness[best_idx])
        generations.append(i+1)

    plot_chart(fitness_list, generations)


def rastringin(chromosome):

    #fitness = 10*len(chromosome)
    fitness = 0

    for i in range(len(chromosome)):
        fitness += (chromosome[i]**2) - (10*math.cos(2*math.pi*chromosome[i]))

    return fitness

def none_function(chromosome):

    return 0;

def plot_chart(best_fitness_axis, generation_axis):
    
    df = pd.DataFrame({'Generations': generation_axis, 
        'Best Fitness': best_fitness_axis})
    
    # multiple line plot
    plt.plot('Generations', 'Best Fitness', data = df, marker='', color='blue', linewidth=6)
    plt.legend()
    plt.show()


it = list(differential_evolution(rastringin, bounds=[(-3, 3), (-3, 3)], use_octave=True))
