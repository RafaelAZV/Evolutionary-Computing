from IPython import embed
import numpy as np
import random
import copy
import pandas as pd
import matplotlib.pyplot as plt 

def read_input(file_name):

	auxiliar_system = []

	with open(file_name, "r") as ins:
	    array = []
	    for line in ins:
	        auxiliar_system.append(line.rstrip('\n').split(" "))
	
	return auxiliar_system

auxiliar_system = read_input("entrada_teste.txt")
def generate_population(population_size, n_orders):

	population = []
	for i in range(0, population_size):
		population.append(random.sample(range(n_orders), n_orders))

	return population

def weighted_random_choice(choices):

    max = sum(choices.values())

    pick = random.uniform(0, max)
    current = 0

    for key, value in choices.items():
        current += value

        if current > pick:
            return key

def parent_selection(population):

	choices = {}

	parent1 = 0
	parent2 = 0

	for individual in range(0, len(population)):
		fit = fitness(population[individual], auxiliar_system)

		if(fit == 0):
			choices[individual] = fit + 1
		else:	
			choices[individual] = fit

	while(parent1 == parent2):
		parent1 = weighted_random_choice(choices)
		parent2 = weighted_random_choice(choices)

	return [copy.deepcopy(population[parent1]), copy.deepcopy(population[parent2])], [parent1, parent2]

def fitness(individual, auxiliar_system):

	total_fitness = 0
	total_fitness = 0
	initial_time = 0
	ranges_map_list = [{}, {}, {}]
	added = False

	for order in individual:
		tasks = auxiliar_system[order]

		for i in range(0, len(tasks)):
		 	if(i == 0):
				if not ranges_map_list[i]:
					ranges_map_list[i][initial_time] = [[int(tasks[i]), initial_time + int(tasks[i])]]
					total_fitness += int(tasks[i])

				else:
					#machineA = ranges_map_list[0]
					#range_map_key_list = machineA.keys()
					last_task_time = ranges_map_list[0][ranges_map_list[0].keys()[-1]][-1][1] + 1
					total_range = last_task_time + int(tasks[i])
					machines_list = [1,2]

					for j in machines_list:

						machine_map = copy.deepcopy(ranges_map_list[j])
						machine_map_key_list = machine_map.keys()
						
						for k in range(last_task_time, total_range):

							if k in machine_map_key_list:

								if(k == last_task_time):
									#iterate here
									for times in machine_map[k]:
										if(times[0] < int(tasks[i])):
											added = True																
											total_fitness += (int(tasks[i]) - times[0])
											ranges_map_list[j][k].append([int(tasks[i]), total_range])
								
								else:
									for times in machine_map[k]:
										added = True
										total_fitness += int(tasks[i]) + times[0] - (total_range - k)
										ranges_map_list[j][k].append([int(tasks[i]), total_range])

						if(not added):
							total_fitness += int(tasks[i])
							ranges_map_list[j][last_task_time] = [[int(tasks[i]), total_range]]

			elif (i == 1):
				if not ranges_map_list[i]:
					auxiliar_time = ranges_map_list[0][0][0][1] + 1
					ranges_map_list[i][auxiliar_time] = [[int(tasks[i]), auxiliar_time + int(tasks[i])]]
					total_fitness += int(tasks[i])

				else:
					last_task_time = ranges_map_list[1][ranges_map_list[1].keys()[-1]][-1][1] + 1
					total_range = last_task_time + int(tasks[i])
					machines_list = [0,2]

					for j in machines_list:

						machine_map = copy.deepcopy(ranges_map_list[j])
						machine_map_key_list = machine_map.keys()
						
						for k in range(last_task_time, total_range):
							if k in machine_map_key_list:

								if(k == last_task_time):
									#iterate here
									for times in machine_map[k]:
										if(times[0] < int(tasks[i])):
											added = True											
											total_fitness += (int(tasks[i]) - times[0])
											ranges_map_list[j][k].append([int(tasks[i]), total_range])
								
								else:
									for times in machine_map[k]:
										added = True
										total_fitness += int(tasks[i]) + times[0] - (total_range - k)
										ranges_map_list[j][k].append([int(tasks[i]), total_range])
						if(not added):
							total_fitness += int(tasks[i])
							ranges_map_list[j][last_task_time] = [[int(tasks[i]), total_range]]

			else:
				if not ranges_map_list[i]:
					auxiliar_time = ranges_map_list[1][auxiliar_time][0][1] + 1
					ranges_map_list[i][auxiliar_time] = [[int(tasks[i]), auxiliar_time + int(tasks[i])]]
					total_fitness += int(tasks[i])

				else:
					last_task_time = ranges_map_list[2][ranges_map_list[2].keys()[-1]][-1][1] + 1
					total_range = last_task_time + int(tasks[i])
					machines_list = [0,1]

					for j in machines_list:

						machine_map = copy.deepcopy(ranges_map_list[j])
						machine_map_key_list = machine_map.keys()
						
						for k in range(last_task_time, total_range):
							if k in machine_map_key_list:
								if(k == last_task_time):
									#iterate here
									for times in machine_map[k]:
										if(times[0] < int(tasks[i])):
											added = True											
											total_fitness += (int(tasks[i]) - times[0])
											ranges_map_list[j][k].append([int(tasks[i]), total_range])
								
								else:
									for times in machine_map[k]:
										added = True
										total_fitness += int(tasks[i]) + times[0] - (total_range - k)
										ranges_map_list[j][k].append([int(tasks[i]), total_range])
						if(not added):
							total_fitness += int(tasks[i])
							ranges_map_list[j][last_task_time] = [[int(tasks[i]), total_range]]
										
	return total_fitness		

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
	
	for i in range(0, 2):
		population.append(children[i])
		population_score.append(children_score[i])

	sorted_population_score = np.argsort(population_score)
	
	indexes = [sorted_population_score[len(sorted_population_score) - 1], sorted_population_score[len(sorted_population_score) - 2]]
	new_population, new_scores = repopulate(copy.deepcopy(population), copy.deepcopy(population_score), copy.deepcopy(indexes))
	
	return new_population, new_scores

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

	population_size = 20
	generations_number = 1000
	children_score = []
	population_score = []
	population = generate_population(20, len(auxiliar_system))

	best_fitness_axis = []
	mean_fitness_axis = []
	generation_axis = []

	for i in range(0, generations_number):

		parents, parents_indexes = parent_selection(copy.deepcopy(population))
		children = ordered_crossover(parents)

		for child in children:
			children_score.append(fitness(child, auxiliar_system))

		for individual in population:
			population_score.append(fitness(individual, auxiliar_system))

		population, population_score = select_individuals(copy.deepcopy(children), copy.deepcopy(population), copy.deepcopy(children_score), copy.deepcopy(population_score))

		best_fitness_axis.append(min(population_score))
		mean_fitness_axis.append(np.mean(population_score))
		generation_axis.append(i)
		population_score = []

	plot_chart(best_fitness_axis, mean_fitness_axis, generation_axis)


if __name__ == '__main__':
	main()