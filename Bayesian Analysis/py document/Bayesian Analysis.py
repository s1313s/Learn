import numpy as np
import matplotlib.pyplot as plt
import emcee

# Setting Graphics Parameters
plt.rcParams['figure.figsize'] = (20, 10)

# Loading Data
ice_data = np.loadtxt('./Bayesian Analysis/dataset/data 2.txt')
age, T = ice_data[:, 2], ice_data[:, 4]

# Defining Model
def model(theta, age=age):
    a1, a2, a3, p1, p2, p3, T0 = theta
    return a1 * np.sin(2 * np.pi * age / p1) + a2 * np.sin(2 * np.pi * age / p2) + a3 * np.sin(
        2 * np.pi * age / p3) + T0

# Defining The Logarithmic Likelihood Function
def lnlike(theta, x, y, yerr):
    return -0.5 * np.sum(((y - model(theta, x)) / yerr) ** 2)

# Defining A Prior Probability Function
def lnprior(theta):
    a1, a2, a3, p1, p2, p3, T0 = theta
    if 0.0 < a1 < 5.0 and 0.0 < a2 < 5.0 and 0.0 < a3 < 5.0 and 10000. < p1 < 200000 and 10000. < p2 < 200000 and 10000. < p3 < 200000 and -10.0 < T0 < 0:
        return 0.0
    return -np.inf

# Defining A Posterior Probability Function
def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

# MCMC Sampling Main Function
def run_mcmc(p0, nwalkers, niter, ndim, lnprob, data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 100)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state

# Randomly Selecting Some Samples From The Sampling Results 
# And Generating A Model
def sample_walkers(nsamples, flattened_chain):
    draw = np.random.choice(len(flattened_chain), size=nsamples, replace=False)
    thetas = flattened_chain[draw]
    models = [model(i) for i in thetas]
    spread = np.std(models, axis=0)
    med_model = np.median(models, axis=0)
    return med_model, spread

# Setting The Data Error And Running The Main Program
Terr = 0.05 * np.mean(T)
data = (age, T, Terr)
nwalkers = 240
niter = 1024
initial = np.array([1.0, 1.0, 1.0, 26000., 41000., 100000., -4.5])
ndim = len(initial)
p0 = [initial + 1e-7 * np.random.randn(ndim) for _ in range(nwalkers)]
sampler, pos, prob, state = run_mcmc(p0, nwalkers, niter, ndim, lnprob, data)

# Sampling Results And Maximum Likelihood Parameters Are Extracted 
# And Graphs Are Generated
flattened_chain = sampler.flatchain
theta_max = flattened_chain[np.argmax(sampler.flatlnprobability)]
best_fit_model = model(theta_max)

med_model, spread = sample_walkers(100, flattened_chain)

plt.plot(age, T, label='Change of Temperature', color='black')
plt.plot(age, best_fit_model, label='Maximum likelihood model', color='blue')
plt.fill_between(age, med_model - spread, med_model + spread, color='green', alpha=0.3, label=r'$1 \sigma$ Posteriori Distribution')

plt.legend()
plt.show()
print(f'Maximum likelihood parameters:\n'
      f'a1: {theta_max[0]:.8f}\n'
      f'a2: {theta_max[1]:.8f}\n'
      f'a3: {theta_max[2]:.8f}\n'
      f'p1: {theta_max[3]:.8f}\n'
      f'p2: {theta_max[4]:.8f}\n'
      f'p3: {theta_max[5]:.8f}\n'
      f'T0: {theta_max[6]:.8f}')

