import numpy as np
import emcee 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import model21cm.error as e

class Model:

    def __init__(self, name=None):
        self.name=name 
        self.globalpriorflag=None
        self.parameters=None
        self.frequencies=None
        self.temperatures=None
        self.weights=None
        self.loglikelihoodflag=None

    def add_parameters(self, parameterlist):
        """Adds parameters and their priors to model."""
        e.parameters_check_1(self) 
        self.parameterlist = parameterlist 
        self.parameters=[]
        for i in parameterlist: 
            self.parameters.append(i.name)

    def globalprior(self, x):
        try: 
            e.parameters_check_2(self) 
        except:
            print(e.ERR3)
        else: 
            value = np.prod(list(map(lambda y, z:y.prior_at(z), self.parameterlist, x)))
            return value

    def add_data(self, data):
        """Adds dataset to model."""
        try:
            e.dataframe_check_1(data)
            e.dataframe_check_2(data)  
        except: 
            print(e.ERR6)
        else: 
            self.frequencies = data.iloc[:,0]
            self.temperatures = data.iloc[:,1]
            if data.shape[1] == 3:
                self.weights = data.iloc[:,2]

    def loglikelihood(self, x, sigma=0.1):
        temps = self.temperatures
        model = forwardmodel(self.frequencies, x, dark=False, mu=0, sig=0.)
        resid_sq = (temps-model)**2.
        chi_sq = np.sum(resid_sq / (sigma**2) ) 
        log_lik = -0.5 * chi_sq
        return log_lik    

    def globalprior_at(self, x): 
        """Returns global prior value at sample point."""
        try:
            e.parameters_check_3(x) 
            e.prior_check_1(self)
        except: 
            print(e.ERR9)
        else:  
            return self.globalprior(x)

    def globallogprior_at(self, x): 
        """Returns log of global prior value at sample point."""
        try:
            e.parameters_check_3(x)
            e.prior_check_1(self) 
        except: 
            print(e.ERR9)
        else: 
            p = self.globalprior(x)
            if p==0:
                return -np.inf
            else:
                return np.log(p) 

    def loglikelihood_at(self, x): 
        """Returns log of global likelihood at sample point."""
        try:
            e.parameters_check_3(x)
            e.data_check_1(self)
        except: 
            print(e.ERR15)
        else: 
            return self.loglikelihood(x)

    def logposterior_at(self, x): 
        """Returns log of global posterior at sample point."""
        try:
            e.parameters_check_3(x)
            e.data_check_1(self)
            e.prior_check_1(self)
        except: 
            print(e.ERR19)
        else: 
            value = self.globallogprior_at(x) + self.loglikelihood_at(x)
            if not np.isfinite(value):
                return -np.inf
            else:
                return value
               
    def MCMC(self,
            start_near=[1565, 650, -1200, 700, -174, 0.53, 6.8, 78.6, 20.7], 
            nwalkers=50, 
            nsteps=600, 
            ncpu=1):
        return generate_mcmc(self, start_near, nwalkers, nsteps, ncpu)

    def __str__(self): 
        return self.name

class DarkModel(Model):  
    
    def __init__(self, name=None): 
        super().__init__(name) #Instantiate Model object with default name=None
        
    def loglikelihood(self, x, sigma = 0.1):
        temps = self.temperatures
        model = forwardmodel(self.frequencies, x, dark=True, mu=0, sig=0.)
        resid_sq = (temps-model)**2.
        chi_sq = np.sum(resid_sq / (sigma**2) )
        log_lik = -0.5 * chi_sq
        return log_lik

    def globalprior_at(self, x): 
        """Returns global prior value at sample point."""
        try:
            e.parameters_check_4(x)
            e.prior_check_1(self)
        except: 
            print(e.ERR22) 
        else: 
            return self.globalprior(x)

    def globallogprior_at(self, x): 
        """Returns log of global prior value at sample point."""
        try:
            e.parameters_check_4(x)
            e.prior_check_1(self) 
        except: 
            print(e.ERR22)
        else: 
            p = self.globalprior(x)
            if p==0:
                return -np.inf
            else:
                return np.log(p) 

    def loglikelihood_at(self, x): 
        """Returns log of global likelihood at sample point."""
        try:
            e.parameters_check_4(x)
            e.data_check_1(self)
        except: 
            print(e.ERR28)
        else: 
            return self.loglikelihood(x)

    def logposterior_at(self, x): 
        """Returns log of global posterior at sample point."""
        try:
            e.parameters_check_4(x)
            e.data_check_1(self)
            e.prior_check_1(self)
        except: 
            print(e.ERR32)
        else: 
            value = self.globallogprior_at(x) + self.loglikelihood_at(x)
            if not np.isfinite(value):
                return -np.inf
            else:
                return value
    
    #starting xi at 0.1 but idk whats best at this point.
    def MCMC(self, 
        start_near = [1565, 650, -1200, 700, -174, 0.53, 6.8, 78.6, 20.7, -0.1],
        nwalkers = 50, 
        nsteps = 600,  
        ncpu=1):
        return generate_mcmc(self, start_near, nwalkers, nsteps, ncpu)
               
               
def foreground(a0, a1, a2, a3, a4, freq): 
    """Returns foreground contribution to model."""
    nu_c = 77 #Verify this
    red_nu = freq/nu_c
    t1 = a0*(red_nu)**(-2.5)
    t2 = a1*(red_nu)**(-1.5)
    t3 = a2*(red_nu)**(-0.5)
    t4 = a3*(red_nu)**(0.5)
    t5 = a4*(red_nu)**(1.5)
    tf = t1 + t2 + t3 + t4 + t5
    return tf

def signal(A, tau, nu0, w, freq):
    """Returns 21-cm signal contribution to model."""
    B = 4*((freq-nu0)**2)*np.log(-1/tau*np.log((1+np.exp(-tau))/2))*(1/w**2)
    top = -A*(1-np.exp(-tau*np.exp(B)))
    bottom = 1-np.exp(-tau)
    T_21 = top/bottom
    return T_21

 
def darksignal(xi,freq):
    """Returns dark matter signal"""
    signal = xi/(freq/77)
    return signal
        
def forwardmodel(freq, theta, dark=False, mu=0, sig=0.1):
	"""Generates data from user input parameter values
	
	Parameters:
	freq: Range of frequencies you consider (array)
	theta: model parameters (tuple)
	"""
	if dark==False:
	    a0, a1, a2, a3, a4, A, tau, nu0, w = theta
	    model = (foreground(a0, a1, a2, a3, a4, freq) 
                    + signal(A, tau, nu0, w, freq)) 
	    noise = np.random.normal(mu, sig, len(freq))
	    forward = model + noise
	    return forward
	else:
	    a0, a1, a2, a3, a4, A, tau, nu0, w, xi = theta
	    model = (foreground(a0, a1, a2, a3, a4, freq) 
                    + signal(A, tau, nu0, w, freq) 
                    + darksignal(xi, freq))
	    noise = np.random.normal(mu, sig, len(freq))
	    forward = model + noise
	    return forward


def generate_mcmc(model = None, 
        start_near = None,
        nwalkers = 50, 
        nsteps = 600, 
        ncpu=1):
        """ Gets MCMC chains using Affine Invariant sampler. 
        
        Parameters:
            nwalkers
            ndim
            nsteps
            start_near: place to start walkers near (list)
            model: instance of Model class
		
        Output:
            sampler object
        """
        if (start_near is not None) and (model is not None):  
            ndim = len(start_near)
            starting_positions = [start_near*(1e-3*np.random.randn(ndim)+1) for i \
                in range(nwalkers)]
            sampler = emcee.EnsembleSampler(nwalkers, ndim, model.logposterior_at, threads=ncpu)
            sampler.run_mcmc(starting_positions, nsteps)
            return sampler
        else: 
            raise Exception(e.ERR33)
		
def check_burn_in(sampler, dark = False):
	"""Shows plot of chains and allows one to fix burn in time"""
	if dark == False:
	
	    fig, (ax_a0, ax_a1, ax_a2, ax_a3, ax_a4, ax_A, ax_tau, ax_nu0, ax_w) = plt.subplots(9)
	    ax_a0.set(ylabel='a0')
	    ax_a1.set(ylabel='a1')
	    ax_a2.set(ylabel='a2')
	    ax_a3.set(ylabel='a3')
	    ax_a4.set(ylabel='a4')
	    ax_A.set(ylabel='A')
	    ax_tau.set(ylabel='tau')
	    ax_nu0.set(ylabel='nu0')
	    ax_w.set(ylabel='w')
	    for i in range(10):
	        df = pd.DataFrame({'a0': sampler.chain[i,:,0], 'a1': sampler.chain[i,:,1], 'a2': sampler.chain[i,:,2], \
	        'a3': sampler.chain[i,:,3], 'a4':sampler.chain[i,:,4],'A': sampler.chain[i,:,5], \
	        'tau':sampler.chain[i,:,6], 'nu0':sampler.chain[i,:,7],'w': sampler.chain[i,:,8]})
	        sns.lineplot(data=df, x=df.index, y='a0', ax=ax_a0)
	        sns.lineplot(data=df, x=df.index, y='a1', ax=ax_a1)
	        sns.lineplot(data=df, x=df.index, y='a2', ax=ax_a2)
	        sns.lineplot(data=df, x=df.index, y='a3', ax=ax_a3)
	        sns.lineplot(data=df, x=df.index, y='a4', ax=ax_a4)
	        sns.lineplot(data=df, x=df.index, y='A', ax=ax_A)
	        sns.lineplot(data=df, x=df.index, y='tau', ax=ax_tau)
	        sns.lineplot(data=df, x=df.index, y='nu0', ax=ax_nu0)
	        sns.lineplot(data=df, x=df.index, y='w', ax=ax_w)
	else: 
	    
	    fig, (ax_a0, ax_a1, ax_a2, ax_a3, ax_a4, ax_A, ax_tau, ax_nu0, ax_w, ax_xi) = plt.subplots(10)
	    ax_a0.set(ylabel='a0')
	    ax_a1.set(ylabel='a1')
	    ax_a2.set(ylabel='a2')
	    ax_a3.set(ylabel='a3')
	    ax_a4.set(ylabel='a4')
	    ax_A.set(ylabel='A')
	    ax_tau.set(ylabel='tau')
	    ax_nu0.set(ylabel='nu0')
	    ax_w.set(ylabel='w')
	    ax_xi.set(ylabel='xi')
	    for i in range(10):
	        df = pd.DataFrame({'a0': sampler.chain[i,:,0], 'a1': sampler.chain[i,:,1], 'a2': sampler.chain[i,:,2], \
	        'a3': sampler.chain[i,:,3], 'a4':sampler.chain[i,:,4],'A': sampler.chain[i,:,5], \
	        'tau':sampler.chain[i,:,6], 'nu0':sampler.chain[i,:,7],'w': sampler.chain[i,:,8], \
	        'xi': sampler.chain[i,:,9]})
	        sns.lineplot(data=df, x=df.index, y='a0', ax=ax_a0)
	        sns.lineplot(data=df, x=df.index, y='a1', ax=ax_a1)
	        sns.lineplot(data=df, x=df.index, y='a2', ax=ax_a2)
	        sns.lineplot(data=df, x=df.index, y='a3', ax=ax_a3)
	        sns.lineplot(data=df, x=df.index, y='a4', ax=ax_a4)
	        sns.lineplot(data=df, x=df.index, y='A', ax=ax_A)
	        sns.lineplot(data=df, x=df.index, y='tau', ax=ax_tau)
	        sns.lineplot(data=df, x=df.index, y='nu0', ax=ax_nu0)
	        sns.lineplot(data=df, x=df.index, y='w', ax=ax_w)
	        sns.lineplot(data=df, x=df.index, y = 'xi', ax = ax_xi)
	 
def trim_to_frame( sampler, burn_in=0, dark = False):
    """
    Chops sampler to only include points sampled after burn in. Returns traces 
    (points sampled by walkers) and Creates dataframe from burned in data. It's useful
    to also return traces because we can use it to make a corner plot.
    
    Input: 
    burn_in: integer
    ndim: integer 
    sampler: sampler object from emcee
    Output:
    parameter_samples: pandas dataframe
    traces: matrix of values sampled by chain. one column for each parameter
    """
    if dark == False:
        samples = sampler.chain[:,burn_in:, :]
        traces = samples.reshape(-1, 9).T
        parameter_samples = pd.DataFrame({'a0': traces[0], 'a1': traces[1], 'a2': traces[2], 'a3': traces[3],\
        'a4': traces[4], 'A': traces[5], 'tau': traces[6], 'nu0': traces[7],\
        'w': traces[8]})
        return traces, parameter_samples
    else:
        samples = sampler.chain[:,burn_in, :]
        traces = samples.reshape(-1,10).T
        parameter_samples = pd.DataFrame({'a0': traces[0], 'a1': traces[1], \
        'a2': traces[2],'a3': traces[3],'a4': traces[4], 'A': traces[5], \
        'tau': traces[6], 'nu0': traces[7],'w': traces[8], 'xi': traces[9]})
        return traces, parameter_samples

