import numpy as np
import math
import matplotlib.pyplot as plt

create_gen = lambda gen_size : [np.random.randint(0,2) for _ in range(gen_size)]

get_gen_size = lambda num : num.bit_length()

get_points = lambda f_range, res : int((f_range/res) + 1)

get_range = lambda min_val, max_val : max_val - min_val

array_to_int = lambda array: "".join(map(str, array))

def fitness(x,y):
    try:
        res = math.sqrt(16*x - x**2 - y**2 - 28) / math.sqrt(-64 + y + 16*x - x**2)
    except (ValueError, ZeroDivisionError):
        res = None
    return res


def get_fenotype(min_val,gen,res):
    gen = array_to_int(gen)
    i = int(gen,2)
    return min_val + (i*res)

def get_fenotypes(individual, min_val_x, min_val_y, resolutions):
    x = get_fenotype(min_val_x, individual[0], resolutions[0])
    y = get_fenotype( min_val_y, individual[1], resolutions[1])
    return (x,y)

def get_population_fenotypes(population, min_val_x, min_val_y, max_val_x, max_val_y, resolutions):
    fenotypes = []
    for individual in population:
        fenotype = get_fenotypes(individual, min_val_x, min_val_y, resolutions)
        fenotypes.append(fenotype)
    for i, fenotype in enumerate(fenotypes):
        if fenotype[0] > max_val_x or fenotype[1] > max_val_y:
            fenotypes.pop(i)
            population.pop(i)
    return [population,fenotypes]

def generate_individual(gen_size_x, gen_size_y):
    gen_x = create_gen(gen_size_x)
    gen_y = create_gen(gen_size_y)
    return [gen_x, gen_y]



create_initial_population = lambda gen_size_x, gen_size_y, population_size: [generate_individual(gen_size_x, gen_size_y) for _ in range(population_size)]

def generate_population(gen_size_x, gen_size_y, min_val_x, max_val_x, min_val_y, max_val_y,resolutions, population_size):
    population = create_initial_population(gen_size_x,gen_size_y, population_size)
    matrix_individual = get_population_fenotypes(population, min_val_x, min_val_y, max_val_x, max_val_y, resolutions)
    if len(matrix_individual[0]) > 1:
        return matrix_individual
    else:
        return generate_population(gen_size_x, gen_size_y, min_val_x,max_val_x, min_val_y, max_val_y, resolutions)

def create_couples(population):
    temp_population = population.copy()
    couples = []
    if (len(temp_population) % 2) != 0:
        temp_population.pop(np.random.randint(0,len(temp_population)-1))
    while len(temp_population) > 0:
        parent_1 = temp_population.pop(np.random.randint(0,len(temp_population)-1))
        if len(temp_population) == 1:
            parent_2 = temp_population.pop(0)
        #elif len(temp_population) <= 0:
        #    return couples.append([parent_1,parent_2])
        else:
            parent_2 = temp_population.pop(np.random.randint(0,len(temp_population)-1))
        couples.append([parent_1,parent_2])
    return couples



def create_sons(couple, population, prob_mutate_individual, prob_mutate_gen):
    parent_1 = couple[0]
    parent_2 = couple[1]

    gen_p1_x = parent_1[0]
    gen_p1_y = parent_1[1]
    gen_p2_x = parent_2[0]
    gen_p2_y = parent_2[1]

    pos_x = np.random.randint(0,len(gen_p1_x)-1)
    pos_y = np.random.randint(0,len(gen_p1_y)-1)

    transfer_gen1_x = gen_p1_x[pos_x:]
    transfer_gen2_x = gen_p2_x[:pos_x]

    transfer_gen1_y = gen_p1_y[pos_y:]
    transfer_gen2_y = gen_p2_y[:pos_y]

    son_1_x = gen_p1_x[pos_x:] + transfer_gen2_x
    son_1_y = gen_p1_y[pos_y:] + transfer_gen2_y

    son_2_x = gen_p2_x[:pos_x] + transfer_gen1_x
    son_2_y = gen_p2_y[:pos_y] + transfer_gen1_y
    
    for gen in [son_1_x, son_1_y, son_2_x, son_2_y]:
        mutate(gen, prob_mutate_individual, prob_mutate_gen )

    population.append([son_1_x, son_1_y])
    population.append([son_2_x, son_2_y])

def mutate(genotype, prob_mutate_individual ,prob_mutate_gen):
    if np.random.uniform(0,1) <= prob_mutate_individual:
        new_gen =  genotype.copy()
        for i,j in enumerate(new_gen):
            if np.random.uniform(0,1) <= prob_mutate_gen:
                if j == 1:
                    new_gen[i] = 0
                else:
                    new_gen[i] = 1
        genotype = new_gen
    return genotype

def evaluate_population(individuals):
    fitness_list = []
    for fenotype in individuals[1]:
        fitness_list.append(fitness(fenotype[0],fenotype[1]))
    return [individuals[0], individuals[1],fitness_list]

def cropping_minimum(individuals):
    index_to_crop = individuals[2].index(min(individuals[2]))
    individuals[0].pop(index_to_crop) 
    individuals[1].pop(index_to_crop) 
    individuals[2].pop(index_to_crop) 

def cropping_maximun(individuals):
    index_to_crop = individuals[2].index(max(individuals[2]))
    individuals[0].pop(index_to_crop) 
    individuals[1].pop(index_to_crop) 
    individuals[2].pop(index_to_crop) 

def get_best_individual(individuals,best_solutions, find_max):
    normalized_fitness = [individual_fitness for individual_fitness in individuals[2] if individual_fitness is not None]
    if find_max:
        index = individuals[2].index(max(normalized_fitness))
    else:
        index = individuals[2].index(min(normalized_fitness))

    best_solutions[0].append(individuals[0][index]) 
    best_solutions[1].append(individuals[1][index])
    best_solutions[2].append(individuals[2][index])


def execute_genetic_algo(min_val_x, min_val_y, max_val_x, max_val_y, res_x, res_y, total_generations, find_max, population_size, prob_mutate_individual, prob_mutate_gen):

    resolutions = [res_x, res_y]

    max_fitness = []
    min_fitness = []
    mean_fitness = []
    best_solutions = [[],[],[]]
    individuals = None
    range_x = get_range(min_val_x,max_val_x)
    num_points_x = get_points(range_x,res_x)
    gen_size_x = get_gen_size(num_points_x)

    range_y = get_range(min_val_y,max_val_y)
    num_points_y = get_points(range_y,res_y)
    gen_size_y = get_gen_size(num_points_y)

    population_and_fenotypes = generate_population(gen_size_x,gen_size_y, min_val_x, max_val_x, min_val_y, max_val_y, resolutions, population_size)

    population = population_and_fenotypes[0]
    individuals = evaluate_population(population_and_fenotypes)


    normalized_fitness = [individual_fitness for individual_fitness in individuals[2] if individual_fitness is not None]
    
    if len(normalized_fitness) >= 1:
        max_fitness.append(max(normalized_fitness))
        min_fitness.append(min(normalized_fitness))
        mean_fitness.append(np.mean(normalized_fitness))
        get_best_individual(individuals,best_solutions, find_max)
        individuals_x = [coordinate[0] for coordinate in individuals[1]]
        individuals_y = [coordinate[1] for coordinate in individuals[1]]
        plt.plot(individuals_x, individuals_y,"o",color="red")
        plt.savefig(f'imgs/0.png')
        plt.title('Generación 0')
    else:
        print('La función no está definida')
    couples = create_couples(population)
    for couple in couples:
        create_sons(couple,population, prob_mutate_individual, prob_mutate_gen)

    individuals = get_population_fenotypes(population, min_val_x, min_val_y, max_val_x, max_val_y, resolutions)
    individuals = evaluate_population(individuals)
    while None in individuals[2]:
            for i, j in enumerate(individuals[2]):
                if j is None:
                    individuals[0].pop(i)
                    individuals[1].pop(i)
                    individuals[2].pop(i)
    individuals_x = [coordinate[0] for coordinate in individuals[1]]
    individuals_y = [coordinate[1] for coordinate in individuals[1]]
    plt.plot(individuals_x, individuals_y,"o",color="red")
    plt.savefig(f'imgs/01.png')
    plt.title('Generación 1')
    for g in range(total_generations):

        individuals = get_population_fenotypes(individuals[0], min_val_x, min_val_y, max_val_x, max_val_y, resolutions)
        individuals = evaluate_population(individuals)
        while None in individuals[2]:
            for i, j in enumerate(individuals[2]):
                if j is None:
                    individuals[0].pop(i)
                    individuals[1].pop(i)
                    individuals[2].pop(i)

        if len(individuals[2]) >= 1:
            print('Ejecutando')
            individuals = get_population_fenotypes(individuals[0], min_val_x, min_val_y, max_val_x, max_val_y, resolutions)
            individuals = evaluate_population(individuals)
            while len(individuals[0]) > population_size:
                if find_max:
                    cropping_minimum(individuals)
                else:
                    cropping_maximun(individuals)
            max_fitness.append(max(individuals[2]))
            min_fitness.append(min(individuals[2]))
            mean_fitness.append(np.mean(individuals[2]))
            get_best_individual(individuals,best_solutions, find_max)
            individuals_x = [coordinate[0] for coordinate in individuals[1]]
            individuals_y = [coordinate[1] for coordinate in individuals[1]]
            plt.plot(individuals_x, individuals_y,"o",color="red")
            if g == 0 or g == 1:
                num_gen = g + 2
            else:
                num_gen = g +1
            plt.title(f'Generación {num_gen}')
            if num_gen < 10:
                name = f'0{num_gen}'
            elif num_gen >= 10 and num_gen <100:
                name = f'{g+1}'
            elif num_gen >= 100 and num_gen < 1000:
                name = f'c_{num_gen}'
            plt.savefig(f'imgs/{name}.png')
            couples = create_couples(individuals[0])
            for couple in couples:
                create_sons(couple,individuals[0], prob_mutate_individual, prob_mutate_gen)
        else:
            print('La función no está definida')

    
    if len(best_solutions[0]) > 0:
        if find_max:
            index = best_solutions[2].index(max(best_solutions[2]))
            title = 'El máximo es: '
            title_graph = 'Evolución del algoritmo en máximo'
            max_case = 'Mejor caso'
            min_case = 'Peor caso'
        else:
            index = best_solutions[2].index(min(best_solutions[2]))
            title = 'El mínimo es: '
            title_graph = 'Evolución del algoritmo en mínimo'
            max_case = 'Peor caso'
            min_case = 'Mejor caso'
        the_best_solution = (best_solutions[0][index],best_solutions[1][index],best_solutions[2][index])
        
        
        x = [num for num in range(len(max_fitness))]
        return (x, max_fitness, mean_fitness, min_fitness, max_case, min_case,the_best_solution,title_graph,title)
    else:
        return None