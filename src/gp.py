import numpy as np
import gpytorch
import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal as TNormal
import seaborn as sns


class GPModel(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, kernel, initialize=True):
        super(GPModel, self).__init__(train_x, train_y, gpytorch.likelihoods.GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
        
        if initialize:
            if type(kernel) == gpytorch.kernels.spectral_mixture_kernel.SpectralMixtureKernel:
                print ('init')
                self.covar_module.initialize_from_data(train_x, train_y)
            elif type(kernel) == gpytorch.kernels.kernel.ProductKernel:
                 for kern in kernel.kernels:
                     if type(kern) == gpytorch.kernels.spectral_mixture_kernel.SpectralMixtureKernel:
                         print ('init')
                         kern.initialize_from_data(train_x, train_y)
        
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    
    
class GP:
    
    def __init__(self, train_x, train_y, kernel, initialize=True):
        self.kernel = kernel
        self.train_x = train_x
        self.train_y = train_y
        self.model = GPModel(self.train_x, self.train_y, self.kernel, initialize)
    

    def sample_f(self, nsamples=1, x=[]):
        if len(x) == 0:
            x = self.train_x
        dist = self.model.forward(x)
        f = np.random.multivariate_normal(dist.mean.detach().numpy(),
                                          dist.covariance_matrix.detach().numpy(),
                                          size=(nsamples))
        return f
    
    
    def plot_cov(self, x=None):
        
        if x is None:
            x = self.train_x
        k = self.model.covar_module(x)
        cov = k.evaluate().detach().numpy()
        #ax, fig = plt.subplots(figsize=(5,5))
        fig = sns.heatmap(cov,cbar=False,xticklabels=False,yticklabels=False,
                          square=True)
        return fig
        
    
    
    def predict(self, test_x):
        
        self.model.eval()
        self.model.likelihood.eval()
        
        with torch.no_grad():
            observed_pred = self.model.likelihood(self.model(test_x))
        return observed_pred
    
    
    def plot(self, test_x=[]):
        
        if len(test_x) == 0:
            test_x = self.train_x
        observed_pred = self.predict(test_x)
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        
        lower, upper = observed_pred.confidence_region()
        ax.plot(self.train_x.numpy(), self.train_y.numpy(), 'k*')
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        ax.fill_between(test_x.numpy(), lower.detach().numpy(), upper.detach().numpy(), alpha=0.5)
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        
    
    def optimize(self, niters=50, lr=0.1):
        
        self.model.train()
        self.model.likelihood.train()
    
        optimizer = torch.optim.Adam([
            {'params': self.model.parameters()},
        ], lr=lr)
        
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        
        for i in range(niters):
            optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -mll(output, self.train_y)
            loss.backward()
            print(loss.item())
            optimizer.step()
            
    
    def optimize_predictive(self, test_x, test_y, niters=50, lr=0.1):
        
        optimizer = torch.optim.Adam([
        {'params': self.model.parameters()},
        ], lr=lr)
    
        for i in range(niters):
            optimizer.zero_grad()
            self.model.eval()
            self.likelihood.eval()
            
            observed_pred = self.likelihood(self.model(test_x))
            loss = -observed_pred.log_prob(test_y)
            
            self.model.train()
            self.likelihood.train()
            loss.backward()
            
            print (loss.item())
            optimizer.step()
            
    
    def optimize_anomalies(self, test_x, test_y, labels, niters=50, lr=0.1):
        
        optimizer = torch.optim.Adam([
        {'params': self.model.parameters()},
        ], lr=lr)
    
        for i in range(niters):
            optimizer.zero_grad()
            self.model.eval()
            self.likelihood.eval()
            
            observed_pred = self.likelihood(self.model(test_x))
            loss = -self.anomaly_log_prob(observed_pred, test_y, labels)
                        
            self.model.train()
            self.likelihood.train()
            loss.backward()
            
            print (loss.item())
            optimizer.step()
    
    
    def anomaly_log_prob(self, observed_pred, value, labels):
        
        mean, covar = observed_pred.loc, observed_pred.lazy_covariance_matrix
        var = covar.diag()
        norm = TNormal(mean, var)
        
        log_prob = norm.log_prob(value)
        prob = torch.exp(log_prob)
        res = (prob**(1-labels)) * ((1-prob)**labels)
        return sum(torch.log(res))