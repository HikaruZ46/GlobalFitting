import numpy as np
from scipy.optimize import minimize, LinearConstraint
from scipy.stats import norm

def fit_global(data, n_components, n_datasets):
    fixed_params = [n_components, data]
    data_flatten = np.concatenate(data).flatten()
    initial_means = np.random.choice(data_flatten, n_components)
    initial_stds = np.full(n_components*n_datasets, np.std(data))
    initial_weights = np.full(n_components*n_datasets, 1/n_components)
    initial_params = np.concatenate([initial_means, initial_stds, initial_weights])
    # weights and stds should be positive
    bounds = [(None, None)]*n_components + [(0, None)]*n_components*n_datasets + [(0, 1)]*n_components*n_datasets
    # add equality constraint for sum of weights
    # sum of weights should be 1
    A_eq = np.zeros((n_datasets, len(initial_params)))
    sum_weights = np.zeros(n_datasets)
    for i in range(n_datasets):
        A_eq[i, (n_datasets+i+1)*n_components:(n_datasets+i+2)*n_components] = 1
        sum_weights[i] = 1
    constraint = LinearConstraint(A_eq, sum_weights, sum_weights)
    result = minimize(objective, initial_params, args=(fixed_params), bounds=bounds, constraints=constraint)
    return result

def objective(params, fixed_params):
    n_components = fixed_params[0]
    datasets = fixed_params[1]
    n_datasets = len(datasets)
    global_means = params[:n_components] # means are fitted global
    local_stds = params[n_components:(n_datasets+1)*n_components] # stds are fitted locally
    local_weights = params[(n_datasets+1)*n_components:] # weights are fitted locally
    log_likelihood = 0
    for i in range(n_datasets):
        data = datasets[i]
        weights = local_weights[i*n_components:(i+1)*n_components]
        means = global_means
        stds = local_stds[i*n_components:(i+1)*n_components]
        ps = gaussian_mixture(np.concatenate([weights, means, stds]), data)
        ps = np.clip(ps, 1e-10, None)  # Avoid log(0) by clipping ps to a minimum value
        log_likelihood += np.sum(np.log(ps))
    return -log_likelihood

def gaussian_mixture(params, x):
    n = len(params) // 3
    weights = params[:n]
    means = params[n:2*n]
    stds = params[2*n:]
    result = np.zeros_like(x)
    for i in range(n):
        result += weights[i] * norm.pdf(x, means[i], stds[i])
    return result

def print_result(x, n_components, n_datasets):
    print('Optimization terminated successfully')
    means = x[:n_components]
    stds = x[n_components:(n_datasets+1)*n_components]
    weights = x[(n_datasets+1)*n_components:]
    print('Means: ', means)
    for i in range(n_datasets):
        print(f'Result Dataset {i+1}')
        print('Std: ', stds[i*n_components:(i+1)*n_components])
        print('Weight: ', weights[i*n_components:(i+1)*n_components])



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data1 = np.concatenate([np.random.normal(0, 1, 1000), np.random.normal(10, 1, 500)])
    data2 = np.concatenate([np.random.normal(0, 1, 500), np.random.normal(10, 1, 1000)])
    n_components = 2
    n_datasets = 2
    data = [data1, data2]
    result = fit_global(data, n_components, n_datasets)
    print_result(result.x, n_components, n_datasets)
    means = result.x[:n_components]
    stds = result.x[n_components:(n_datasets+1)*n_components]
    weights = result.x[(n_datasets+1)*n_components:]
    x = np.linspace(min(data1), max(data1), 1000)
    for k in range(n_datasets):
        plt.hist(data[k], bins=50, alpha=0.5, density=True)
        for i in range(n_components):
            plt.plot(x, weights[k*n_datasets+i]*norm.pdf(x, means[i], stds[k*n_datasets+i]), label=f'Component {i+1}', linestyle='--')
        plt.legend()
        plt.show()