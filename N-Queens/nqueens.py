import numpy as np
import random 
from IPython import embed
import matplotlib.pyplot as plt
import pandas as pd
import sys

def initialize(queens_number, individuals_number):

	population = []

	for i in range(0, individuals_number):
		genotipes = random.sample(range(queens_number), queens_number)
		population.append(genotipes)
	
	return population

def fitness_util(genotipe):

	colisions_number = 0
	length = len(genotipe)
	for i in range(0, length):
		for j in range(0, length):
			if((abs(i-j) == abs(genotipe[i] - genotipe[j])) and (i != j)):
				colisions_number += 1
	
	return colisions_number/2


def fitness(population):

	population_score = []
	for i in population:
		population_score.append(fitness_util(i))

	return population_score

def select_parents(population):

	random_parents = []
	random_parents_score = []
	for i in range(0, 5):
		index = np.random.randint(low = 0, high = len(population))
		random_parents.append(population[index])
		random_parents_score.append(fitness_util(population[index]))
		sorted_indexes = np.argsort(random_parents_score)
	
	return [population[sorted_indexes[len(sorted_indexes) - 1]], population[sorted_indexes[len(sorted_indexes) - 2]]]


def repopulate(population, population_score, indexes):

	new_population = []
	new_scores = []
	index1 = indexes[0]
	index2 = indexes[1]

	for i in range(0, len(population_score)):
		if(i != index1 and i != index2):
			new_population.append(population[i])
			new_scores.append(population_score[i])
	
	return new_population, new_scores


def select_individuals(children, population, children_score, population_score):
	
	for i in range(0, len(children_score)):
		population.append(children[i])
		population_score.append(children_score[i])

	sorted_population_score = np.argsort(population_score)
	
	indexes = [sorted_population_score[len(sorted_population_score) - 1], sorted_population_score[len(sorted_population_score) - 2]]
	new_population, new_scores = repopulate(population, population_score, indexes)
	
	return new_population, new_scores


def check_genotipes(child, parent, child_inherited, start, end, size):

	current_parent_position = 0
	fixed_position = list(range(start, end + 1))       
	i = 0
	while i < size:

		if i in fixed_position:
			i += 1
			continue

		check_child = child[i]
		if check_child == -1: #to be filled

			parent_trait = parent[current_parent_position]
			while parent_trait in child_inherited:

				current_parent_position += 1
				parent_trait = parent[current_parent_position]

			child[i] = parent_trait
			child_inherited.append(parent_trait)

		i +=1

	return child


def mutation(genotipe):

	index1 = -1
	index2 = -1

	while(index1 == index2):
		index1 = np.random.randint(low = 0, high = len(genotipe))
		index2 = np.random.randint(low = 0, high = len(genotipe))

	auxiliar = genotipe[index1]
	genotipe[index1] = genotipe[index2]
	genotipe[index2] = auxiliar

	return genotipe


def ordered_crossover(parents):

	parent1 = parents[0]
	parent2 = parents[1]

	size = len(parent1)

	child1, child2 = [-1] * size, [-1] * size
	start, end = sorted([random.randrange(size) for _ in range(2)])

	child1_inherited = []
	child2_inherited = []

	for  i in range(start, end  + 1):
		child1[i] = parent1[i]
		child2[i] = parent2[i]
		child1_inherited.append(parent1[i])
		child2_inherited.append(parent2[i])
	
	child1 = check_genotipes(child1, parent2, child1_inherited, start, end, size)
	child2 = check_genotipes(child2, parent1, child2_inherited, start, end, size)

	if(random.random() < 0.8):
		child1 = mutation(child1)
		child2 = mutation(child2)

	return [child1, child2]

def check_optimal(population_score):

	for i in population_score:
		if int(i) == 0:
			return True

	return False

def plot_chart(best_fitness_axis, mean_fitness_axis, generation_axis):

	df = pd.DataFrame({'Generations': generation_axis, 
		'Best Fitness': best_fitness_axis, 
		'Mean Fitness': mean_fitness_axis})
	

	# multiple line plot
	plt.plot('Generations', 'Best Fitness', data = df, marker='', color='blue', linewidth=6)
	plt.plot('Generations', 'Mean Fitness', data = df, marker='', color='red', linewidth=2)
	plt.legend()
	plt.show()


def main():

	max_iterations = 5000
	generations_number = 0
	queens_number = int(sys.argv[1])
	individuals_number = int(sys.argv[2])
	population = initialize(queens_number, individuals_number)
	population_score = fitness(population)

	generation_axis = []
	best_fitness_axis = []
	mean_fitness_axis = []

	while((generations_number < max_iterations)):

		parents = select_parents(population)
		children = ordered_crossover(parents)
		children_score = fitness(children)
		population, population_score = select_individuals(children, population, children_score, population_score)
		best_fitness_axis.append(min(population_score))
		mean_fitness_axis.append(np.mean(population_score))
		generation_axis.append(generations_number)
		generations_number += 1
		print(generations_number)
		if(check_optimal(population_score) == True):
			break

	plot_chart(best_fitness_axis, mean_fitness_axis, generation_axis)

if __name__ == '__main__':
	main()
