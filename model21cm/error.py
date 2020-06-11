import pandas as pd

def parameters_check_1(model): 
    if model.parameters is not None:
        raise RuntimeWarning(ERR1)

def parameters_check_2(model): 
    if model.parameters==None:
        raise Exception(ERR2)

def parameters_check_3(parameters):
    if len(parameters) is not 9:
        raise ValueError(ERR7)

def parameters_check_4(parameters):
    if len(parameters) is not 10:
        raise ValueError(ERR20)

def dataframe_check_1(data):
    if type(data) is not pd.core.frame.DataFrame:
        raise ValueError(ERR4)

def dataframe_check_2(data): 
    if (data.shape[1] is not 2) and (data.shape[1] is not 3):
        raise ValueError(ERR5)

def prior_check_1(model): 
    if model.globalprior is None: 
        raise Exception(ERR8)

def data_check_1(model): 
    if model.frequencies is None: 
        raise Exception(ERR14)

ERR1 = "Caution, the parameters for this model were previously set. Those \
values have now been overwritten as well as the corresponding global prior."

ERR2 = "Priors on parameters not set!"

ERR3 = "Can't give global prior. Priors on parameters not set!"

ERR4 = "Input data must be a pandas dataframe."

ERR5 = "Input data must contain only 2 or 3 data columns."

ERR6 = "Improper data format provided - the input data must be a pandas \
dataframe and contain exactly 2 or exacly 3 data columns."

ERR7 =  "This model has exactly 9 parameters, you must specify a 9 element \
list of the sample value in the order: a0, a1, a2, a3, a4, A, tau, nu0, w."

ERR8 = "No parameters have been added to the model which contain a prior \
distribution."

ERR9 = "This model has exactly 9 parameters, to evaluate the global prior at \
a point in parameter space, you must first provide parameters with priors to \
the model, then specify a 9 element list of the sample value in the order: \
a0, a1, a2, a3, a4, A, tau, nu0, w."

ERR14 = "No frequency/temperature data has been added to the model. You must \
first add such data before the global likelihood can be evaluated."

ERR15 = "This model has exactly 9 parameters, to evaluate the global \
likelihood at a point in parameter space, you must first provide a dataset \
to the model, then specify a 9 element list of the sample value in the order: \
a0, a1, a2, a3, a4, A, tau, nu0, w."

ERR19 = "This model has exactly 9 parameters, to evaluate the global \
posterior at a point in parameter space, you must first provide a dataset and \
parameters with prior information to the model, then specify a 9 element list \
of the sample value in the order: a0, a1, a2, a3, a4, A, tau, nu0, w."

ERR20 = "This model has exactly 10 parameters, you must specify a 10 element \
list of the sample value in the order: a0, a1, a2, a3, a4, A, tau, nu0, w, xi."

ERR22 = "This model has exactly 10 parameters, to evaluate the global prior \
at a point in parameter space, you must first provide parameters with priors \
to the model, then specify a 10 element list of the sample value in the \
order: a0, a1, a2, a3, a4, A, tau, nu0, w, xi."

ERR28 = "This model has exactly 10 parameters, to evaluate the global \
likelihood at a point in parameter space, you must first provide a dataset \
to the model, then specify a 10 element list of the sample value in the \
order: a0, a1, a2, a3, a4, A, tau, nu0, w, xi."

ERR32 = "This model has exactly 10 parameters, to evaluate the global \
posterior at a point in parameter space, you must first provide a dataset and \
parameters with prior information to the model, then specify a 10 element list \
of the sample value in the order: a0, a1, a2, a3, a4, A, tau, nu0, w, xi."

ERR33 = "Must specify appropriate model and starting position for walkers in \
parameter space."
