import torch
from scipy.stats import kstest, norm, beta, expon

import numpy as np
import json

class normalmix():
    
    def __init__(self, *args):
        
        self.normals = []
        self.wts = []
        
        for i in range(len(args)//3):
            self.normals.append(norm(args[3*i+1],args[3*i+2]))
            self.wts.append(args[3*i])
            
    def cdf(self, arg):
        
        cdf_vals = []
        
        for wt,normal in zip(self.wts,self.normals):
            cdf_vals.append(wt*normal.cdf(arg))
        return sum(cdf_vals)
        
        
        
def is_tol(a, b):
    if type(a) == dict:
        keys_match = (set(a) == set(b))
        if keys_match:
            for k,v in a.items(): #check all items
                if not is_tol(v, b[k]): #recursively check if they match
                    return False 
            return True #return True if all items match
        else:
            return False #false if keys don't match
    else:
        return not torch.any(torch.logical_not(torch.abs((a - b)) < 1e-5))




def run_prob_test(stream, truth, num_samples):
    samples = []
    for i in range(int(num_samples)):
        samples.append(next(stream))

    print(samples[0])
    
    distrs = {
            'normal' : norm,
            'beta' : beta,
            'exponential' : expon,
            'normalmix' : normalmix,
            }
    
    print(truth)
    truth_dist = distrs[truth[0]](*truth[1:])

    d,p_val = kstest(np.array(samples), truth_dist.cdf)
    
    return p_val
    
def load_truth(path): # sorry this is very hacky, and will break for anything complicated
    with open(path) as f:            
        truth = json.load(f)
    if type(truth) is list:
        if type(truth[0]) is str:
            truth = tuple(truth)
        else:
            truth = torch.tensor(truth)
    if type(truth) is dict:
        truth = {float(k):v for k,v in truth.items()} ##TODO: this will NOT work for nested dicts
    return truth

