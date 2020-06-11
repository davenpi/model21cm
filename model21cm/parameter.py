import pandas as pd
import numpy as np
import model21cm as m21


#Class to hold properties of a particular parameter
class Parameter: 


    def __init__(self, name=None, uniform_min=None, uniform_max=None): 
        self.name=name
        self.uniformpriorflag=None
        self.jeffreyspriorflag=None
        self.jeffreys_min = None
        self.jeffreys_max = None 
        self.uniform_min = uniform_min
        self.uniform_max = uniform_max
        if (self.uniform_min is not None) and (self.uniform_max is not None): 
            if self.uniform_max < self.uniform_min:
                raise ValueError("Error: second input must be larger than first.")
                self.uniform_min=None
                self.uniform_max=None
            else:    
                self.uniformpriorflag=True

    def prior(self, x):
        if self.uniformpriorflag == True: 
            if x<self.uniform_min or x>self.uniform_max:
                return 0.
            else:
                return 1. / (self.uniform_max - self.uniform_min)
        elif self.jeffreyspriorflag ==True:    
            if x<self.jeffreys_min or x>self.jeffreys_max:
                return 0. 
            else: 
                return (1. / x) * (1. / np.log(self.jeffreys_max / self.jeffreys_min))     
        else: 
            raise Exception("No prior set for parameter.")        

    def set_uniform_prior(self, parameter_min, parameter_max):
        if parameter_max > parameter_min: 
           self.jeffreyspriorflag = False
           self.uniformpriorflag = True
           self.uniform_min = parameter_min
           self.uniform_max = parameter_max
        else: 
            raise ValueError("Error: Invalid choice of max/min for Uniform Prior.")

    def set_jeffreys_prior(self, parameter_min, parameter_max):
        if parameter_min>0 and parameter_max>0 and parameter_max>parameter_min: 
           self.jeffreyspriorflag = True
           self.uniformpriorflag = False
           self.jeffreys_min = parameter_min
           self.jeffreys_max = parameter_max
        else: 
            raise ValueError("Error: Invalid choice of max/min for Jeffreys Prior.")

    #redundant function 
    def prior_at(self, x): 
        return self.prior(x)

    def logprior_at(self, x):
        """Returns value of log of prior at x."""
        p = self.prior(x)
        if p==0: 
            return -np.inf
        else: 
            return np.log(p)


#Class to hold multiple instances the `parameter` class. 
class Parameters: 

    def __init__(self, model_name=None, parameter_list=[]):
        self.model_name = model_name
        self.parameter_list = []
        self.add_parameter(parameter_list)

    def add_parameter(self, parameter_list): 
        if isinstance(parameter_list, m21.Parameter): 
            parameter_list = [parameter_list]
        for p in parameter_list: 
            if p not in self.parameter_list: 
                self.parameter_list.append(p)

    def remove_parameter(self, parameter_list): 
        if isinstance(parameter_list, m21.Parameter): 
            parameter_list = [parameter_list]
        for p in parameter_list: 
            self.parameter_names.remove(p.name)
            self.parameter_list.remove(p)

    def parameter_names(self):  
        return list(map(lambda y:y.name, self.parameter_list))
