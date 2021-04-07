from evaluator import evaluate
import torch
import numpy as np
import json
import sys
from daphne import daphne
import pickle

def run_until_observe_or_end(res):
    cont, args, sigma = res
    res = cont(*args)
    while type(res) is tuple:
        if res[2]['type'] == 'observe':
            return res
        cont, args, sigma = res
        res = cont(*args)

    res = (res, None, {'done' : True}) #wrap it back up in a tuple, that has "done" in the sigma map
    return res

def resample_particles(particles, log_weights):
    n_particles = len(particles)
    normalized_probs = torch.tensor(log_weights).exp() / torch.tensor(log_weights).exp().sum()
    sample_indices = torch.multinomial(input=normalized_probs, num_samples=n_particles, replacement=True)

    new_particles = []
    for index in sample_indices:
        new_particles.append(particles[index])
    logZ = torch.tensor(log_weights).exp().sum().log() 
    logZ += torch.tensor(1/n_particles).log()

    return logZ, new_particles



def SMC(n_particles, exp):

    particles = []
    weights = []
    logZs = []
    output = lambda x: x

    for i in range(n_particles):

        res = evaluate(exp, env=None)('addr_start', output)
        logW = 0.
        particles.append(res)
        weights.append(logW)

    #can't be done after the first step, under the address transform, so this should be fine:
    done = False
    smc_cnter = 0
    while not done:
        print('In SMC step {}, Zs: '.format(smc_cnter), logZs)
        for i in range(n_particles): #Even though this can be parallelized, we run it serially
            res = run_until_observe_or_end(particles[i])
            if 'done' in res[2]: #this checks if the calculation is done
                particles[i] = res[0]
                if i == 0:
                    done = True  #and enforces everything to be the same as the first particle
                    address = ''
                else:
                    if not done:
                        raise RuntimeError('Failed SMC, finished one calculation before the other')
            else:
                if i == 0:
                    init_address = res[2]['alpha']
                particle_address = res[2]['alpha']
                assert init_address == particle_address

                particles[i] = res
                weights[i] = res[2]['logW']

        if not done:
            #resample and keep track of logZs
            logZn, particles = resample_particles(particles, weights)
            logZs.append(logZn)
        smc_cnter += 1
    logZ = sum(logZs)
    return logZ, particles


if __name__ == '__main__':

    for i in range(1,5):
        exp = daphne(['desugar-hoppl-cps', '-i', '../CS532-HW6/programs/{}.daphne'.format(i)])

        logZ_values = []
        for particle_power in range(6):
            n_particles = 10**particle_power
            print("Program", i, "evaluating", n_particles, "particles.")
            logZ, particles = SMC(n_particles, exp)

            logZ_values += [logZ]
            values = torch.stack(particles)

            with open('Daphne_program' + str(i) + '_n_particles_' + str(n_particles) + '_values.pickle', 'wb+') as f:
                pickle.dump(values, f)

        with open('Daphne_program' + str(i) + '_n_particles_' + str(n_particles) + 'logZ_values.pickle', 'wb+') as f:
            pickle.dump(logZ_values, f)
