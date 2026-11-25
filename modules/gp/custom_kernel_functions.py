import torch
from gpytorch.kernels import MaternKernel, ScaleKernel, RBFKernel, Kernel, rbf_kernel
from botorch.utils.transforms import unnormalize
import math
from gpytorch.functions import MaternCovariance
from gpytorch.settings import trace_mode


class Linear_Regression(Kernel):

    has_lengthscale = True

    def __init__(self,
                 bounds,
                 fA_function,
                 raw_fA_lengthscale=0.0,
                 Y_train_mean=None,
                 Y_train_std=None,
                 rbf=None,
                 nu=None,
                 **kwargs):

        if nu:
            if nu not in {0.5, 1.5, 2.5}:
                raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        
        super(Linear_Regression, self).__init__(**kwargs)
        self.nu = nu
        self.bounds=bounds
        self.fA_function=fA_function
        self.register_parameter("raw_fA_lengthscale",torch.nn.Parameter(torch.tensor([raw_fA_lengthscale],dtype=torch.double)))
        self.rbf=rbf
        self.Y_train_mean=Y_train_mean
        self.Y_train_std=Y_train_std

    @property
    def fA_lengthscale(self):
        return torch.nn.functional.softplus(self.raw_fA_lengthscale)


    def forward(self, x1_init, x2_init, diag=False, **params):
        x1_init=x1_init.to(torch.double)
        x2_init=x2_init.to(torch.double)


        x1_init_unormalized=unnormalize(x1_init,self.bounds)
        x1_init_unormalized=x1_init_unormalized.to(torch.double)

        x2_init_unormalized=unnormalize(x2_init,self.bounds)
        x2_init_unormalized=x2_init_unormalized.to(torch.double)

        Y_train_mean=-self.Y_train_mean
        Y_train_std=-self.Y_train_std

        fA_x1=self.fA_function.fA(x1_init_unormalized)
        fA_x1=fA_x1.to(torch.double)
        fA_x1=(fA_x1-Y_train_mean)/Y_train_std
        fA_x1=fA_x1.to(torch.double)

        fA_x2=self.fA_function.fA(x2_init_unormalized)
        fA_x2=fA_x2.to(torch.double)
        fA_x2=(fA_x2-Y_train_mean)/Y_train_std
        fA_x2=fA_x2.to(torch.double)

        fA_distance = self.covar_dist(fA_x1.div(self.fA_lengthscale), fA_x2.div(self.fA_lengthscale), diag=diag, **params)
        fA_distance=fA_distance.to(torch.double)

        x_distance=self.covar_dist(x1_init.div(self.lengthscale),x2_init.div(self.lengthscale))
        x_distance=x_distance.to(torch.double)

        if self.nu:
            fA_exp_component = torch.exp(-math.sqrt(self.nu * 2) * fA_distance)
            x_exp_component = torch.exp(-math.sqrt(self.nu * 2) * x_distance)
                        
            if self.nu == 0.5:
                fA_constant_component = 1
                x_constant_component=1
            elif self.nu == 1.5:
                fA_constant_component = (math.sqrt(3) * fA_distance).add(1)
                x_constant_component= (math.sqrt(3) * x_distance).add(1)
            elif self.nu == 2.5:
                fA_constant_component = (math.sqrt(5) * fA_distance).add(1).add((5.0 / 3.0) * fA_distance**2)
                x_constant_component = (math.sqrt(5) * x_distance).add(1).add((5.0 / 3.0) * x_distance**2)


            fA_custom_component=fA_constant_component * fA_exp_component
            x_component=x_constant_component*x_exp_component

            return x_component*fA_custom_component

        if self.rbf:
            fA_custom_component=torch.exp((-fA_distance**2)/2)
            x_component=torch.exp((-x_distance**2)/2)
            return fA_custom_component*x_component




class Only_fA_Linear_Regression(Kernel):

    has_lengthscale = True

    def __init__(self,
                 bounds,
                 fA_function,
                 rbf=None,
                 nu=None,
                 Y_train_mean=None,
                 Y_train_std=None,
                 **kwargs):

        if nu:
            if nu not in {0.5, 1.5, 2.5}:
                raise RuntimeError("nu expected to be 0.5, 1.5, or 2.5")
        
        super().__init__(**kwargs)
        self.nu = nu
        self.bounds=bounds
        self.fA_function=fA_function
        self.rbf=rbf
        self.Y_train_mean=Y_train_mean
        self.Y_train_std=Y_train_std


    def forward(self, x1_init, x2_init, diag=False, **params):
        x1_init=x1_init.to(torch.double)
        x2_init=x2_init.to(torch.double)

        x1_init_unormalized=unnormalize(x1_init,self.bounds)
        x1_init_unormalized=x1_init_unormalized.to(torch.double)

        x2_init_unormalized=unnormalize(x2_init,self.bounds)
        x2_init_unormalized=x2_init_unormalized.to(torch.double)

        Y_train_mean=-self.Y_train_mean
        Y_train_std=-self.Y_train_std
        

        fA_x1=self.fA_function.fA(x1_init_unormalized)
        fA_x1=fA_x1.to(torch.double)
        fA_x1=(fA_x1-Y_train_mean)/Y_train_std
        fA_x1=fA_x1.to(torch.double)

        fA_x2=self.fA_function.fA(x2_init_unormalized)
        fA_x2=fA_x2.to(torch.double)
        fA_x2=(fA_x2-Y_train_mean)/Y_train_std
        fA_x2=fA_x2.to(torch.double)

        

        fA_distance = self.covar_dist(fA_x1.div(self.lengthscale), fA_x2.div(self.lengthscale), diag=diag, **params)
        fA_distance=fA_distance.to(torch.double)


        if self.nu:
            fA_exp_component = torch.exp(-math.sqrt(self.nu * 2) * fA_distance)
                        
            if self.nu == 0.5:
                fA_constant_component = 1
            elif self.nu == 1.5:
                fA_constant_component = (math.sqrt(3) * fA_distance).add(1)
            elif self.nu == 2.5:
                fA_constant_component = (math.sqrt(5) * fA_distance).add(1).add((5.0 / 3.0) * fA_distance**2)


            fA_custom_component=fA_constant_component * fA_exp_component

            return fA_custom_component

        if self.rbf:
            fA_custom_component=torch.exp((-fA_distance**2)/2)
            return fA_custom_component




