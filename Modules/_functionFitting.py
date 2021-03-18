import numpy as np

def log_likelihood(model, y, y_err):
    # Commented section for log_f 
    sigma2 = y_err**2 #+ model**2 * np.exp(2*log_f)

    return -0.5 * np.sum((y-model)**2 / sigma2 + np.log(2*np.pi*sigma2))

def log_prior(parameters, parameters_limit):
    n_p = np.size(parameters)
    check = np.zeros(n_p)
    for i in np.arange(np.size(parameters)):
        if parameters_limit[i][0]< parameters[i] < parameters_limit[i][1]:
            check[i] = True
        else:
            check[i] = False
    
    if np.all(check):
        return 0.0

    return -np.inf

def log_probability(parameters, parameters_limit, model, y, y_err):
    lp = log_prior(parameters, parameters_limit)

    if not np.isfinite(lp):
        return -np.inf

    #log_f = parameters[-1]
    #return lp + log_likelihood(model, y, y_err, log_f)
    return lp + log_likelihood(model, y, y_err)



