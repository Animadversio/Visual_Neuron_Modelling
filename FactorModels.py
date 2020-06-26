"""
Define a factorized convolution model for the neurons

tensorflow 1*1 convolution and linearized fullly connected layer

"""

from numpy import exp, sin
from scipy.optimize import leastsq


def residual(variables, x, data, eps_data):
    """Model a decaying sine wave and subtract data."""
    amp = variables[0]
    phaseshift = variables[1]
    freq = variables[2]
    decay = variables[3]

    model = amp * sin(x*freq + phaseshift) * exp(-x*x*decay)

    return (data-model) / eps_data


variables = [10.0, 0.2, 3.0, 0.007]
out = leastsq(residual, variables, args=(x, data, eps_data))


#%%

def FactorModel(variables, x, data, eps_data):
    """Model a decaying sine wave and subtract data."""
    amp = variables[0]
    phaseshift = variables[1]
    freq = variables[2]
    decay = variables[3]

    model = amp * sin(x*freq + phaseshift) * exp(-x*x*decay)

    return (data-model) / eps_data


variables = [10.0, 0.2, 3.0, 0.007]
out = leastsq(residual, variables, args=(x, data, eps_data))


