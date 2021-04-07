import torch
import torch.distributions as dist



class Normal(dist.Normal):
    
    def __init__(self, loc, scale):
        
        if scale > 20.:
            self.optim_scale = scale.clone().detach().float().requires_grad_()
        else:
            self.optim_scale = torch.log(torch.exp(scale.float()) - 1).clone().detach().requires_grad_()
        
        
        super().__init__(loc.float(), torch.nn.functional.softplus(self.optim_scale))

    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.loc, self.optim_scale]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        ps = [p.clone().detach().requires_grad_() for p in self.Parameters()]
         
        return Normal(*ps)
    
    def log_prob(self, x):
        
        self.scale = torch.nn.functional.softplus(self.optim_scale)
        
        return super().log_prob(x)
        
class Bernoulli(dist.Bernoulli):
    
    def __init__(self, probs):
        if type(probs) is float:
            probs = torch.tensor(probs)
        logits = torch.log(probs/(1-probs)) ##will fail if probs = 0
        #
        super().__init__(logits = logits)

    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.logits]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        logits = [p.clone().detach().requires_grad_() for p in self.Parameters()][0]
        
        return Bernoulli(logits = logits)
    
class Categorical(dist.Categorical):
    
    def __init__(self, probs):
        
        probs = probs / probs.sum(-1, keepdim=True)
        logits = dist.utils.probs_to_logits(probs)

        logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        super().__init__(logits = logits)

        self.logits = logits.clone().detach().requires_grad_()
        self._param = self.logits

    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.logits]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        logits = [p.clone().detach().requires_grad_() for p in self.Parameters()][0]
        
        return Categorical(logits = logits)    

class Dirichlet(dist.Dirichlet):
    
    def __init__(self, concentration):
        #NOTE: logits automatically get added
        super().__init__(concentration)

    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.concentration]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        concentration = [p.clone().detach().requires_grad_() for p in self.Parameters()][0]
        
        return Dirichlet(concentration)

class Gamma(dist.Gamma):
    
    def __init__(self, concentration, rate):
        #NOTE: logits automatically get added
        super().__init__(concentration, rate)
    
    def Parameters(self):
        """Return a list of parameters for the distribution"""
        return [self.concentration, self.rate]
        
    def make_copy_with_grads(self):
        """
        Return a copy  of the distribution, with parameters that require_grad
        """
        
        concentration,rate = [p.clone().detach().requires_grad_() for p in self.Parameters()]
        
        return Gamma(concentration, rate)
