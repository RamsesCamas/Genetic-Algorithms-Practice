import numpy as np
import matplotlib.pyplot as plt
"""First definitive attemp on createing a genetic algorithm from scratch
and without the use of external libreries.
In this example the number of generations defined is 10,000 plus
the zero generation"""

create_gen = lambda gen_size : np.random.randint(0,2,(gen_size))

get_gen_size = lambda num : num.bit_length()

get_points = lambda f_range, res : int((f_range/res) + 1)

get_range = lambda min_val, max_val : max_val - min_val

POPULATION_SIZE = 100
PROB_MUTATE_GEN = 0.1
PROB_MUTATE_INDIVIDUAL = 0.15

def fitness(x):
    return (x/2)*np.cos((np.pi*x)/2)

def get_fenotype(min_val,gen,res):
    gen = array_to_int(gen)
    i = int(gen,2)
    return min_val + (i*res)

def array_to_int(array):
    return "".join(map(str, array))

def create_initial_population(gen_size):
    population = []
    for _ in range(POPULATION_SIZE):
        population.append(create_gen(gen_size))
    return population

def get_fenotypes(population,min_val, max_val,res):
    fenotypes = []
    for individual in population:
        fenotype = get_fenotype(min_val,individual,res)
        if fenotype <= max_val:
            fenotypes.append(fenotype)
        else:
            population = [ x for x in population if not (x==individual).all()]
    return [population,fenotypes]

def mutate(genotype):
    if np.random.uniform(0,1) <= PROB_MUTATE_INDIVIDUAL:
        new_gen =  genotype.copy()
        for i,j in enumerate(new_gen):
            if np.random.uniform(0,1) <= PROB_MUTATE_GEN:
                if j == 1:
                    new_gen[i] = 0
                else:
                    new_gen[i] = 1
        genotype = new_gen
    return genotype

def generate_population(gen_size,min_val, max_val,res):
    population = create_initial_population(gen_size)
    tuple_individual = get_fenotypes(population,min_val,max_val,res)
    if  len(tuple_individual[0]) > 1:
        return tuple_individual
    else:
        return generate_population(gen_size,min_val,max_val,res)

def create_matches(population):
    temp_population = population.copy()
    couples = []
    if len(temp_population) % 2 !=0:
        temp_population.pop(np.random.randint(0,len(temp_population)-1))
    while len(temp_population) > 0:
        parent_1 = temp_population.pop(np.random.randint(0,len(temp_population)-1))
        if len(temp_population) == 1:
            parent_2 = temp_population.pop(0)
        else:
            parent_2 = temp_population.pop(np.random.randint(0,len(temp_population)-1))
        couples.append([parent_1,parent_2])
    return couples

def create_sons(couple,population):
    parent_1 = couple[0]
    parent_2 = couple[1]
    gen_p1 = parent_1
    gen_p2 = parent_2
    pos = np.random.randint(0,len(gen_p1)-1)
    transfer_gen1 = gen_p1[pos:]
    transfer_gen2 = gen_p2[:pos]
    son_1 = np.concatenate((gen_p1[pos:],transfer_gen2))
    son_2 = np.concatenate((gen_p2[:pos],transfer_gen1))
    mutate(son_1)
    mutate(son_2)
    population.append(son_1)
    population.append(son_2)

def evaluate_population(individuals):
    fitness_list = []
    for fenotype in individuals[1]:
        fitness_list.append(fitness(fenotype))
    return [individuals[0], individuals[1],fitness_list]

def cropping(individuals):
    index_to_crop = individuals[2].index(min(individuals[2]))
    print('Eliminando ', individuals[2][index_to_crop]) 
    individuals[0].pop(index_to_crop) 
    individuals[1].pop(index_to_crop) 
    individuals[2].pop(index_to_crop) 
    return index_to_crop

def get_best_individual(individuals,best_solutions):
    max_index = individuals[2].index(max(individuals[2]))
    best_solutions[0].append(individuals[0][max_index]) 
    best_solutions[1].append(individuals[1][max_index])
    best_solutions[2].append(individuals[2][max_index])

if __name__ == '__main__':
    min_val = 4
    max_val = 10
    res = 0.01
    max_fitness = []
    min_fitness = []
    mean_fitness = []
    best_solutions = [[],[],[]]

    f_range = get_range(min_val,max_val)
    num_points = get_points(f_range,res)
    gen_size = get_gen_size(num_points)

    individuals = generate_population(gen_size,min_val,max_val,res)
    individuals = evaluate_population(individuals)
    
    max_fitness.append(max(individuals[2]))
    min_fitness.append(min(individuals[2]))
    mean_fitness.append(np.mean(individuals[2]))
    get_best_individual(individuals,best_solutions)
    
    couples = create_matches(individuals[0])
    for couple in couples:
        create_sons(couple,individuals[0])

    individuals = get_fenotypes(individuals[0],min_val,max_val,res)
    individuals = evaluate_population(individuals)
    get_best_individual(individuals,best_solutions)

    for _ in range(1000):
        while len(individuals[0]) > POPULATION_SIZE:
            cropping(individuals)
            
        individuals = get_fenotypes(individuals[0],min_val,max_val,res)
        individuals = evaluate_population(individuals)
        
        max_fitness.append(max(individuals[2]))
        min_fitness.append(min(individuals[2]))
        mean_fitness.append(np.mean(individuals[2]))
        get_best_individual(individuals,best_solutions)

        couples = create_matches(individuals[0])
        for couple in couples:
            create_sons(couple,individuals[0])

    maximum_index = best_solutions[2].index(max(best_solutions[2]))
    the_best_solution = (best_solutions[0][maximum_index],best_solutions[1][maximum_index],best_solutions[2][maximum_index])
    print('El máximo es: ', the_best_solution)
    print(len(max_fitness))
    print(max_fitness[-1])
    plt.style.use('_mpl-gallery')
    x = [num for num in range(1001)]
    plt.plot(x,max_fitness)
    plt.plot(x,mean_fitness)
    plt.plot(x,min_fitness)
    plt.show()