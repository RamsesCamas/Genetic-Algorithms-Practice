import numpy as np

create_gen = lambda gen_size : np.random.randint(0,2,(gen_size))

get_gen_size = lambda num : num.bit_length()

get_points = lambda f_range, res : int((f_range/res) + 1)

get_range = lambda min_val, max_val : max_val - min_val

POPULATION_SIZE = 4
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
    return (population,fenotypes)

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
    population.append(son_1)
    population.append(son_2)

def evaluate_population(individuals):
    fitness_list = []
    for fenotype in individuals[1]:
        fitness_list.append(fitness(fenotype))
    return (individuals[0], individuals[1],fitness_list)

if __name__ == '__main__':
    min_val = 4
    max_val = 10
    res = 0.01
    max_fitness = []
    min_fitness = []
    mean_fitness = []

    f_range = get_range(min_val,max_val)
    num_points = get_points(f_range,res)
    gen_size = get_gen_size(num_points)

    individuals = generate_population(gen_size,min_val,max_val,res)

    first_fitness = evaluate_population(individuals)[2]
    max_fitness.append(max(first_fitness))
    min_fitness.append(min(first_fitness))
    mean_fitness.append(np.mean(first_fitness))

    print(len(individuals[0]))
    couples = create_matches(individuals[0])
    for couple in couples:
        create_sons(couple,individuals[0])
    for individual in individuals[0]:
        mutate(individual)