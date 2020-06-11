from .io import Data
from .model import Model, DarkModel, foreground, signal, darksignal, forwardmodel, generate_mcmc, check_burn_in, trim_to_frame
from .parameter import Parameter, Parameters

name = "model21cm"
edgesdata = Data('edges_data.csv')
