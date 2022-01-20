from collections import namedtuple
from functools import partial
from random import choices, randint, randrange, uniform
from time import time
from typing import Callable, List, Tuple

#Implementation of genetic algorithm based on this video: https://youtu.be/nhT56blfRpE 
#From the Youtune Channel "Kie Codes"

Genome = List[int]
Population = List[Genome]
Thing = namedtuple('Thing',['name','value','weight'])

FitnessFunc = Callable[[Genome],int]
PopulateFunc = Callable[[],Population]
SelectionFunc = Callable[[Population,FitnessFunc], Tuple[Genome,Genome]]
CrossoverFunc = Callable[[Genome,Genome],Tuple[Genome,Genome]]
MutationFunc = Callable[[Genome],Genome]

things = [
    Thing('Laptop',500,2200),
    Thing('Headphones',150,160),
    Thing('Notepad',40,333),
    Thing('Water Bottle',30,192),
    Thing('Coffe Mug',60,350),
]

def generate_genome(lenght: int) -> Genome:
    return choices([0,1],k=lenght)

def generate_population(size:int,genome_lenght:int)-> Population:
    return [generate_genome(genome_lenght) for _ in range(size)]

def fitness(genome:Genome, things: List[Thing], weight_limit: int) -> int:
    if len(genome) != len(things):
        raise ValueError("genome and thins must be of the same lenght")
    weight = 0
    value = 0
    for i, thing in enumerate(things):
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value
            if weight > weight_limit:
                return 0
    return value  

def selection_pair(population:Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k=2
    )

def single_point_crossover(a:Genome, b:Genome) -> Tuple[Genome,Genome]:
    if len(a) != len(b):
        raise  ValueError('Genomes a and b must be of same lenght')

    lenght = len(a)
    if lenght < 2:
        return a,b

    p = randint(1,lenght-1)
    return a[0:p] + b[p:], b[0:p] +a[p:]

def mutation(genome: Genome, num: int=1, probability: float=0.5) -> Genome:
    for _ in range(num):
        index = randrange(len(genome))
        genome[index] = genome[index] if uniform(0,1) > probability else abs(genome[index] -1)
    return genome

def run_evolution(
    populate_func: PopulateFunc,
    fitness_func: FitnessFunc,
    fitness_limit: int,
    generation_limit: int = 100,
    selection_func: SelectionFunc = selection_pair,
    crossover_func: CrossoverFunc = single_point_crossover,
    mutation_func: MutationFunc = mutation
)-> Tuple[Population,int]:

    population = populate_func()
    for i in range(generation_limit):
        population = sorted(population,key=lambda genome: fitness_func(genome), reverse=True)
        if fitness_func(population[0]) >= fitness_limit:
            break
        next_generation = population[0:2]
        for _ in range((len(population)//2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a,offspring_b]
        population = next_generation
    population = sorted(population,key=lambda genome: fitness_func(genome), reverse=True)
    return population, i

def genome_to_things(genome: Genome, things: List[Thing]) -> List[Thing]:
    result = []
    for i, thing in enumerate(things):
        if genome[i] == 1:
            result += [thing.name]
    return result

if __name__ == '__main__':
    populate_func = partial(generate_population,size=10,genome_lenght=len(things))
    fitness_func = partial(fitness, things=things, weight_limit=3000)
    fitness_limit = 740
    genration_limit = 100
    start_algorithm = time()
    population, generations = run_evolution(populate_func,fitness_func,fitness_limit,genration_limit)
    end_algorithm = time()
    print(f'Number of generations {generations}')
    print(f'Best solution: {genome_to_things(population[0],things)}')
    print(f'Time of execution {end_algorithm-start_algorithm}')