#bo_loop
import torch
from botorch.sampling.stochastic_samplers import StochasticSampler
from botorch.utils.transforms import unnormalize,normalize
from botorch import fit_gpytorch_mll
from modules.sumo import check_feasibility
from botorch.acquisition import qLogExpectedImprovement,LogExpectedImprovement
from torch.quasirandom import SobolEngine
import numpy as np
import pandas as pd
import traceback

class BOLoop():
    def __init__(self,
                 niter,
                 init_df,
                 output_dir,
                 bounds,
                 gp_initializer,
                 objective_function,
                 optimizer,
                 dtype,
                 x_dim,
                 loss_scale
                 ):
        
        self.niter=niter
        self.init_df=init_df
        self.output_dir=output_dir
        self.bounds=bounds
        self.gp_initializer=gp_initializer
        self.objective_function=objective_function
        self.optimizer=optimizer
        self.dtype=dtype
        self.x_dim=x_dim
        self.loss_scale=loss_scale
    
    def run_loop(self,
                 kernel_name,
                 annotation,
                 rs,
                 num_replications,
                 mean_name=None,
                 demand_scaling_factor=None
                 ):
        

        
            print(f"{kernel_name}")
            if mean_name:
                print(f"{mean_name}")
            print("-----------------------------------------------")

            hyperparams_dict={
                "raw_lengthscale":[],
                "lengthscale":[],
                "raw_amplitude":[],
                "amplitude":[],
                "noise":[],
                "raw_noise":[],
                "constant_mean":[],
                "fA_lengthscale":[],
                "raw_fA_lengthscale":[]
                }
            

            for i in range(self.niter):
                
                print(f"####### {i+1} ########")
                
                df = self.init_df if i == 0 else df
                                
                train_X = torch.from_numpy(df[['gt_0','gt_1','gt_2','gt_3','gt_4','gt_5','gt_6']].values)
                train_X_norm = normalize(train_X,self.bounds)
                train_X_norm=train_X_norm.to(self.dtype)
                
                train_Y = -torch.from_numpy(df[["loss_estimate"]].values) 
                train_Y=train_Y.to(self.dtype)

                Y_train_mean=train_Y.mean(dim=0).item()
                Y_train_std=train_Y.std(dim=0).item()
                
                best_y,best_y_index = torch.max(train_Y,dim=0)
                best_y=best_y.item()
                best_y_index=best_y_index.item()

                gp_model, gp_mll = self.gp_initializer.initialize_gp_model(train_X=train_X_norm,
                                                                           train_Y=train_Y,
                                                                           kernel_name=kernel_name,
                                                                           Y_train_mean=Y_train_mean,
                                                                           Y_train_std=Y_train_std,
                                                                           mean_name=mean_name)



                if i==0:
                    hyperparams_dict["raw_lengthscale"].append(gp_model.covar_module.base_kernel.raw_lengthscale)
                    hyperparams_dict["lengthscale"].append(gp_model.covar_module.base_kernel.lengthscale)
                    hyperparams_dict["amplitude"].append(gp_model.covar_module.outputscale)
                    hyperparams_dict["raw_amplitude"].append(gp_model.covar_module.raw_outputscale)
                    hyperparams_dict["noise"].append(gp_model.likelihood.noise)
                    hyperparams_dict["raw_noise"].append(gp_model.likelihood.noise_covar.raw_noise)
                    if "linear" not in mean_name:
                        hyperparams_dict["constant_mean"].append(gp_model.mean_module.constant)
                    if (kernel_name!="RBF") and (not "default" in kernel_name) and (not "only" in kernel_name):
                        hyperparams_dict["fA_lengthscale"].append(gp_model.covar_module.base_kernel.fA_lengthscale)
                        hyperparams_dict["raw_fA_lengthscale"].append(gp_model.covar_module.base_kernel.raw_fA_lengthscale)

                try:
                    fit_gpytorch_mll(gp_mll)
                    hyperparams_dict["raw_lengthscale"].append(gp_model.covar_module.base_kernel.raw_lengthscale)
                    hyperparams_dict["lengthscale"].append(gp_model.covar_module.base_kernel.lengthscale)
                    hyperparams_dict["amplitude"].append(gp_model.covar_module.outputscale)
                    hyperparams_dict["raw_amplitude"].append(gp_model.covar_module.raw_outputscale)
                    hyperparams_dict["noise"].append(gp_model.likelihood.noise)
                    hyperparams_dict["raw_noise"].append(gp_model.likelihood.noise_covar.raw_noise)
                    if "linear" not in mean_name:
                        hyperparams_dict["constant_mean"].append(gp_model.mean_module.constant)
                    if (kernel_name!="RBF") and (not "default" in kernel_name) and (not "only" in kernel_name):
                        hyperparams_dict["fA_lengthscale"].append(gp_model.covar_module.base_kernel.fA_lengthscale)
                        hyperparams_dict["raw_fA_lengthscale"].append(gp_model.covar_module.base_kernel.raw_fA_lengthscale)
                
                
                except Exception as e:
                    print("\n--- GP DIAGNOSTICS ---")

                    print("train_X shape:", train_X_norm.shape)
                    print("train_Y shape:", train_Y.shape)

                    print("X dtype:", train_X_norm.dtype)
                    print("Y dtype:", train_Y.dtype)

                    print("X NaN:", torch.isnan(train_X_norm).any())
                    print("Y NaN:", torch.isnan(train_Y).any())

                    print("X Inf:", torch.isinf(train_X_norm).any())
                    print("Y Inf:", torch.isinf(train_Y).any())

                    print("Y mean:", train_Y.mean().item())
                    print("Y std:", train_Y.std().item())

                    print("Unique X:", torch.unique(train_X_norm, dim=0).shape[0])

                    print("Fitting failed:")
                    print(type(e).__name__, str(e))
                    traceback.print_exc()
                    raise RuntimeError("Could not fit the GP model") from e
                
                if mean_name is not None:
                    epoch_index=f"BO_iter_{i}__{kernel_name}_{mean_name}_{annotation}_restart{rs}"
                else:
                    epoch_index=f"BO_iter_{i}__{kernel_name}_{annotation}_restart{rs}"

                EI = LogExpectedImprovement(gp_model, best_f=best_y)
                
                # Maximize acquisition function to get next observation
                new_x = self.optimizer.optimize_acqf_and_get_observation(acq_func=EI,bounds=self.bounds)
                print("------------------------------")
                print(new_x)
                print("------------------------------")
                # if the new observation is not feasible, then replace w/ a randomly sampled point
                if not check_feasibility(new_x.cpu().detach().numpy().reshape(-1)):
                    print('*** BO optimal point is not feasible, replacing w/ a randomly sampled point')
                    is_feasible = False
                    sobol = SobolEngine(self.x_dim, scramble=True,seed=rs)
                    while not is_feasible:
                        x1 = sobol.draw(1).to(dtype=self.dtype)
                        # map the normalized into the original parameter space
                        new_x = unnormalize(x1, self.bounds)
                        if check_feasibility(new_x.cpu().detach().numpy().reshape(-1)):
                            is_feasible = True                        

                train_new_x=new_x.cpu().detach().numpy().reshape(-1)
                new_obj = self.objective_function.compute_error(g_vec=train_new_x,epoch_index=epoch_index,num_replications=num_replications,demand_scaling_factor=demand_scaling_factor)
                
                new_row=pd.DataFrame(np.append(train_new_x,[new_obj,i]).reshape(1,-1),columns=[f"gt_{i}" for i in range(self.x_dim)]+["loss_estimate","iter"])
            
                df = pd.concat([df, new_row], axis=0, ignore_index=True)

            
            if mean_name is not None:
                torch.save(hyperparams_dict,f"./hp_logs/hp_dict_{kernel_name}_{mean_name}_{annotation}_restart{rs}.pt")
                df.to_csv(f'{self.output_dir}BO_df_{kernel_name}_{mean_name}_{annotation}_restart{rs}.csv', index=None)
            else:
                torch.save(hyperparams_dict,f"./hp_logs/hp_dict_{kernel_name}_{annotation}_restart{rs}.pt")
                df.to_csv(f'{self.output_dir}BO_df_{kernel_name}_{annotation}_restart{rs}.csv', index=None)
                