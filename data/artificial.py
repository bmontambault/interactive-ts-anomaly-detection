import torch
import gpytorch
import numpy as np
import pandas as pd
from PIL import Image

import sys
sys.path.append("../src")
from gp import GP

import matplotlib.pyplot as plt


def load_image(image, save=None):
    
    img = Image.open(image)
    img.load()
    data = np.asarray(img, dtype="int32")
    data = 1 - data.sum(axis=2).argmin(axis=0)
    if save != None:
        fig, ax = plt.subplots()
        ax.plot(data,'k-')
        fig.savefig(save)
    return data
        
        
def sample_function(kernel, nsamples, n_fsamples, initialize=True):
    
    train_x = torch.linspace(0, 1, nsamples)
    train_y = torch.linspace(0, 1, nsamples)
    gp = GP(train_x, train_y, kernel, initialize)
    fsamples = gp.sample_f(n_fsamples)
    
    
    return gp, fsamples


def rq_samples(log_lengthscale=-2.5, log_alpha=0, nsamples=100, n_fsamples=11):
    
    kernel = gpytorch.kernels.RQKernel(log_lengthscale=log_lengthscale,
                                       log_alpha=log_alpha)
    gp, fsamples = sample_function(kernel, nsamples, n_fsamples)
    return gp, fsamples
    

def sm_lin_samples(num_mixtures=2,
                   log_mixture_weights=torch.from_numpy(np.array([[0, 2]])).float(),
                   log_mixture_means=torch.from_numpy(np.array([[0, 0]])).float(),
                   log_mixture_scales=torch.from_numpy(np.array([[[[5]], [[0]]]])).float(),
                   variance=1, offset=0,
                   nsamples=100, n_fsamples=11,
                   initialize=False):
    
    lin_kern = gpytorch.kernels.LinearKernel(1, variance=variance,
                                             offset=offset)
    sm_kern = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=2,
        log_mixture_weights=log_mixture_weights,
        log_mixture_means=log_mixture_means,
        log_mixture_scales=log_mixture_scales)
    
    kernel = lin_kern * sm_kern
    gp, fsamples = sample_function(kernel, nsamples, n_fsamples, initialize=initialize)
    return gp, fsamples



#gp, fsamples = rq_samples()




"""
i = pd.read_csv('sm_i.csv').values
i = (i - i.mean()) / i.std()
i = i - i[0]

j = pd.read_csv('sm_j.csv').values
j = (j - j.mean()) / j.std()
j = j - j[0]

k = pd.read_csv('sm_k.csv').values
k = (k - k.mean()) / k.std()
k = k - k[0]
"""
"""
data = j
x = torch.from_numpy(np.linspace(0,2,len(data))).float()
y = torch.from_numpy(data.ravel()).float()

gp = GP(x, y, gpytorch.kernels.SpectralMixtureKernel(num_mixtures=2)*gpytorch.kernels.LinearKernel(1))
gp.optimize(100)
fsamples1 = gp.sample_f(10)

#plt.plot(fsamples1[0])
plt.plot(fsamples1[0])
plt.show()
gp.plot_cov()
"""

"""
log_mixture_weights=torch.from_numpy(np.array([[-1.5985,  0.9900]])).float()
log_mixture_means=torch.from_numpy(np.array([[-3.5516, -5.3902]])).float()
log_mixture_scales=torch.from_numpy(np.array([[[[ 1.0403]],[[-3.9297]]]])).float()
gp, fsamples = sm_lin_samples(2, log_mixture_weights, log_mixture_means, log_mixture_scales,
                          variance=1.5578, offset=1.1347, initialize=False)
plt.plot(fsamples.T)
plt.show()
gp.plot_cov()
"""
"""
log_mixture_weights=torch.from_numpy(np.array([[-1.5985,  10.9900]])).float()
log_mixture_means=torch.from_numpy(np.array([[-3.5516, -5.3902]])).float()
log_mixture_scales=torch.from_numpy(np.array([[[[ 1.0403]],[[-3.9297]]]])).float()
gp, fsamples = sm_lin_samples(2, log_mixture_weights, log_mixture_means, log_mixture_scales,
                          variance=1, offset=0, initialize=False)
plt.plot(fsamples.T)
plt.show()
gp.plot_cov()
"""

"""
log_mixture_weights=torch.from_numpy(np.array([[ 0.3277, -3.5473, 1.]])).float()
log_mixture_means=torch.from_numpy(np.array([[2.3542, 0.9220, -2]])).float()
log_mixture_scales=torch.from_numpy(np.array([[[[ 0.1303]],[[-2.7316]], [[.1]]]])).float()
gp,fsamples = sm_lin_samples(3, log_mixture_weights, log_mixture_means, log_mixture_scales,
                          variance=1, offset=0, initialize=False)
plt.plot(fsamples.T)
plt.show()
gp.plot_cov()
"""

"""
log_mixture_weights=torch.from_numpy(np.array([[2, 0]])).float()
log_mixture_means=torch.from_numpy(np.array([[0, 0]])).float()
log_mixture_scales=torch.from_numpy(np.array([[[[0]], [[10]]]])).float()
gp, fsamples = sm_lin_samples(2, log_mixture_weights, log_mixture_means, log_mixture_scales,
                          variance=1, offset=0, initialize=False)
plt.plot(fsamples.T)
plt.show()
gp.plot_cov()
"""

"""
log_mixture_weights=torch.from_numpy(np.array([[0, 2]])).float()
log_mixture_means=torch.from_numpy(np.array([[0, 0]])).float()
log_mixture_scales=torch.from_numpy(np.array([[[[5]], [[0]]]])).float()
gp, fsamples = sm_lin_samples(2, log_mixture_weights, log_mixture_means, log_mixture_scales,
                          variance=.1, offset=0, initialize=False)
plt.plot(fsamples.T)
plt.show()
gp.plot_cov()
"""
