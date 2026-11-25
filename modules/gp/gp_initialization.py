from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms import Standardize
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, Kernel, rbf_kernel
from gpytorch.means import ConstantMean,ZeroMean
from gpytorch.constraints import Interval
from .custom_kernel_functions import Linear_Regression,Only_fA_Linear_Regression
from .custom_mean_functions import CustomMean
import torch


class InitializeGPModel():

    def __init__(self,
                 fa_function,
                 bounds,
                 dtype=torch.double,
                 rbf=True,
                 mean_mod=True,
                 nu=None,
                 ):
        
        self.fa_function=fa_function
        self.bounds=bounds
        self.mean_mod=mean_mod
        self.dtype=dtype
        self.rbf=rbf
        self.nu=nu

    def initialize_gp_model(self,train_X,train_Y, kernel_name, Y_train_mean, Y_train_std, mean_name=None):
        
        dim = train_X.size(dim=1)

        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))

        if kernel_name == 'RBF_default':
            covar_module = ScaleKernel( RBFKernel() )
            
        elif kernel_name == 'MaternKernel_default':
            covar_module = ScaleKernel( MaternKernel(nu=self.nu) ) 

        elif kernel_name == "linear_regression":
            covar_module=ScaleKernel(Linear_Regression(bounds=self.bounds,fA_function=self.fa_function,dtype=self.dtype,rbf=self.rbf,nu=self.nu,Y_train_mean=Y_train_mean,Y_train_std=Y_train_std))
            
        elif kernel_name=="only_fA_linear_regression":
            covar_module=ScaleKernel(Only_fA_Linear_Regression(bounds=self.bounds,fA_function=self.fa_function,dtype=self.dtype,rbf=self.rbf,nu=self.nu,Y_train_mean=Y_train_mean,Y_train_std=Y_train_std))
        else:
            raise TypeError('Error non-defined kernel choice')


        if mean_name=="ConstantMean":
            mean_module=ConstantMean()
        elif mean_name=="linear_regression":
            mean_module = CustomMean(bounds=self.bounds,fa_function=self.fa_function,Y_train_mean=Y_train_mean,Y_train_std=Y_train_std)
        else:
            mean_module=None   
        

        if (self.mean_mod) and (mean_module is not None):
            gp_model = SingleTaskGP(
                train_X, train_Y, 
                mean_module=mean_module,
                covar_module=covar_module, likelihood=likelihood, 
                outcome_transform=Standardize(m=1)
            )
        else:
            gp_model = SingleTaskGP(
                train_X, train_Y, 
                covar_module=covar_module, likelihood=likelihood, 
                outcome_transform=Standardize(m=1)
            )

        gp_model.covar_module.base_kernel.lengthscale=0.6931
        if mean_name=="ConstantMean":
            gp_model.mean_module.constant=0
        gp_model.covar_module.outputscale=0.6931
        

        gp_mll = ExactMarginalLogLikelihood(gp_model.likelihood, gp_model)
        
        return gp_model, gp_mll