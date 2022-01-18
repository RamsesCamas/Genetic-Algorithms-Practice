import numpy as np

#This was the first approach for a genetic algorithm from scratch
#and only from teory

class Individual():
    def __init__(self,gen=None) -> None:
        self.genotype = gen
        self.fenotype = None
        self.fitness  = None

    def __str__(self) -> str:
        return f"{self.get_genotype()}"

    def set_genotype(self, gen):
        self.genotype = gen
    def set_fenotype(self, fenotype):
        self.fenotype = fenotype
    def set_fitness(self, fit):
        self.fitness = fit

    def get_genotype(self):
        return self.genotype
    def get_fenotype(self):
        return self.fenotype
    def get_fitness(self):
        return self.fitness

POPULATION_SIZE = 4

def fitness(x):
    return (x/2)*np.cos((np.pi*x)/2)

get_gen_size = lambda num : num.bit_length()

get_points = lambda f_range, res : int((f_range/res) + 1)

get_range = lambda min_val, max_val : max_val - min_val

def get_fenotype(pos_val,gen,res):
    gen = array_to_int(gen)
    i = int(gen,2)
    return pos_val + (i * res)

create_gen = lambda gen_size : np.random.randint(0,2,(gen_size))


def create_population(gen_size):
    population = []
    for _ in range(POPULATION_SIZE):
        new_individual = Individual()
        new_individual.set_genotype(create_gen(gen_size))
        population.append(new_individual)
    return population

def array_to_int(array):
    return "".join(map(str, array))

def create_matches(population):
    temp_population = population.copy()
    couples = []
    while len(temp_population) > 0:
        parent_1 = temp_population.pop(np.random.randint(0,len(temp_population)-1))
        if len(temp_population) == 1:
            parent_2 = temp_population.pop(0)
        else:
            parent_2 = temp_population.pop(np.random.randint(0,len(temp_population)-1))
        couples.append([parent_1,parent_2])
    return couples

PROB_MUTATE_GEN = 0.1
PROB_MUTATE_INDIVIDUAL = 0.15

def mutate(individual):
    new_gen =  individual.get_genotype().copy()
    for i,j in enumerate(new_gen):
        if np.random.uniform(0,1) <= PROB_MUTATE_GEN:
            if j == 1:
                new_gen[i] = 0
            else:
                new_gen[i] = 1
    individual.set_genotype(new_gen)
    return individual

def create_sons(couple,population):
    parent_1 = couple[0]
    parent_2 = couple[1]
    gen_p1 = parent_1.get_genotype()
    gen_p2 = parent_2.get_genotype()
    pos = np.random.randint(0,len(gen_p1)-1)
    transfer_gen1 = gen_p1[pos:]
    transfer_gen2 = gen_p2[:pos]
    gen_son_1 = np.concatenate((gen_p1[pos:],transfer_gen2))
    gen_son_2 = np.concatenate((gen_p2[:pos],transfer_gen1))
    son_1 = Individual(gen_son_1)
    son_2 = Individual(gen_son_2)
    population.append(son_1)
    population.append(son_2)

if __name__ == '__main__':
    min_val = 4
    max_val = 10
    f_range = get_range(min_val, max_val)
    res = 0.01
    total_points = get_points(f_range,res)
    gen_size = get_gen_size(total_points)
    population = create_population(gen_size)
    couples = create_matches(population)
    test_couple = couples[0]
    print('Población Gen 0')
    for i in population:
        print(i)
    for couple in couples:
        create_sons(couple,population)
    print('Población Gen 1 (sin poda ni mutación)')
    for i in population:
        print(i)
    print('Población Gen 1 (con mutación, sin poda)')
    for i in population:
        if np.random.uniform(0,1) <= PROB_MUTATE_INDIVIDUAL:
            #print('Antes de mutar')
            #print(i)
            mutate(i)
            #print('Después de mutar')
            #print(i)
        else: print(i)