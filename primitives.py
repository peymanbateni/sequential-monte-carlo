import torch
import torch.distributions as tdist
import distributions as dist

def Normal(alpha, loc, scale, k):
    return k(dist.Normal(loc.float(), scale.float()))

def Bernoulli(alpha, probs, k):
    return k(dist.Bernoulli(probs))

def Categorical(alpha, probs, k):
    return k(dist.Categorical(probs=probs))

def Dirichlet(alpha, concentration, k):
    return k(dist.Dirichlet(concentration))

def Gamma(alpha, concentration, rate, k):
    return k(dist.Gamma(concentration, rate))

def Beta(alpha, arg0, arg1, k):
    return k(tdist.Beta(arg0, arg1))

def Exponential(alpha, rate, k):
    return k(tdist.Exponential(rate))

def Uniform(alpha, hi, lo, k):
    return k(tdist.Uniform(hi, lo))



def sqrt(alpha, arg, k):
    return k(torch.sqrt(arg.float()))

def exp(alpha, arg, k):
    return k(torch.exp(arg.float()))

def log(alpha, arg, k):
    return k(torch.log(arg.float()))

def tanh(alpha, arg, k):
    return k(torch.tanh(arg.float()))

def add(alpha, a, b, k):
    return k(torch.add(a, b))

def mul(alpha, a, b, k):
    return k(torch.mul(a,b))

def div(alpha, a, b, k):
    return k(torch.div(a,b))

def sub(alpha, a, b, k):
    return k(torch.sub(a,b))

def gt(alpha, a, b, k):
    return k(torch.gt(a, b))

def lt(alpha, a, b, k):
    return k(torch.lt(a,b))


def vector(alpha, *args):
    k = args[-1]
    args = args[:-1]
    if len(args) == 0:
        return k(torch.tensor([]))
    elif type(args[0]) is torch.Tensor:
        try:
            output = torch.stack(args) #stack works for 1D, but also ND
        except Exception:
            output = list(args) #NOTE:  that these are NOT persistent
        return k(output)
    else:
        return k(list(args)) #this is for probability distributions


def hashmap(alpha, *args):
    k = args[-1]
    args = args[:-1]
    new_map = {} #NOTE: also not persistent
    for i in range(len(args)//2):
        if type(args[2*i]) is torch.Tensor:
            key = args[2*i].item()
        elif type(args[2*i]) is str:
            key = args[2*i]
        else:
            raise ValueError('Unkown key type, ', args[2*i])
        new_map[key] = args[2*i+1]
    return k(new_map)

def first(alpha, sequence, k):
    return k(sequence[0])

def second(alpha, sequence, k):
    return k(sequence[1])

def rest(alpha, sequence, k):
    return k(sequence[1:])


def last(alpha, sequence, k):
    return k(sequence[-1])

def get(alpha, data, element, k):
    if type(data) is dict:
        if type(element) is torch.Tensor:
            key = element.item()
        elif type(element) is str:
            key = element
        return k(data[key])
    else:
        return k(data[int(element)])

def put(alpha, data, element, value, k): #vector, index, value
    if type(data) is dict:
        newhashmap = data.copy() #NOTE: right now we're copying
        if type(element) is torch.Tensor:
            key = element.item()
        elif type(element) is str:
            key = element
        newhashmap[key] = value
        return k(newhashmap)
    else:
        newvector = data.clone() 
        newvector[int(element)] = value
        return k(newvector)

def remove(alpha, data, element, k):
    if type(data) is dict:
        newhashmap = data.copy()
        if type(element) is torch.Tensor:
            key = element.item()
        elif type(element) is str:
            key = element
        _ = newhashmap.pop(key)        
        return k(newhashmap)
    else:
        idx = int(element)
        newvector = torch.cat([data[0:idx],data[idx+1:]],dim=0)
        return k(newvector)
    
def append(alpha, data, value, k):
    return k(torch.cat([data,torch.tensor([value])], dim=0))

def is_empty(alpha, arg, k):
    return k(len(arg) == 0)

def peek(alpha, sequence, k): #NOTE: only defined for vector
    return k(sequence[0])

def conj(alpha, sequence, element, k):
    if type(sequence) is torch.Tensor:
        return k(torch.cat((element.reshape(1), sequence)))
    elif type(sequence) is list:
        return k([element] + sequence)


def mat_transpose(alpha, arg, k):
    return k(torch.transpose(arg, 1, 0))

def mat_mul(alpha, arg0, arg1, k):
    return k(torch.matmul(arg0,arg1))
    
def mat_repmat(alpha, mat, dim, n, k):
    shape = [1,1]
    shape[int(dim)] = int(n)
    return k(mat*torch.ones(tuple(shape)))


def push_addr(alpha, value, k):
    # print('pushing ', value, ' onto ', alpha)
    return k(alpha + '_' + value)


env = {
       #distr
           'normal': Normal,
           'beta': Beta,
           'discrete': Categorical,
           'dirichlet': Dirichlet,
           'exponential': Exponential,
           'uniform-continuous': Uniform,
           'gamma': Gamma,
           'flip': Bernoulli,
           
           # #math
           'sqrt': sqrt,
           'exp': exp,
           'log': log,
           'mat-tanh' : tanh,
           'mat-add' : add,
           'mat-mul' : mat_mul,
           'mat-transpose' : mat_transpose,
           'mat-repmat' : mat_repmat,
           '+': add,
           '-': sub,
           '*': mul,
           '/': div,
           
           # #
           '<' : lt,
           '>' : gt,
           # '<=' : torch.le,
           # '>=' : torch.ge,
           # '=' : torch.eq,
           # '!=' : torch.ne,
           # 'and' : torch.logical_and,
           # 'or' : torch.logical_or,

           'vector': vector,
           'hash-map' : hashmap,
           'get': get,
           'put': put,
           'append': append,
           'first': first,
           'second': second,
           'rest': rest,
           'last': last,
           'empty?': is_empty,
           'conj' : conj,
           'peek' : peek,

           'push-address' : push_addr,
           }


