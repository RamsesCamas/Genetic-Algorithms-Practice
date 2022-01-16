import numpy as np

create_gen = lambda gen_size : np.random.randint(0,2,(gen_size))

get_gen_size = lambda num : num.bit_length()

get_points = lambda f_range, res : int((f_range/res) + 1)

get_range = lambda min_val, max_val : max_val - min_val

def fitness(x):
    return (x/2)*np.cos((np.pi*x)/2)

def get_fenotype(min_val,gen,res):
    gen = array_to_int(gen)
    i = int(gen,2)
    return min_val + (i*res)

def array_to_int(array):
    return "".join(map(str, array))