from botorch.optim import optimize_acqf
import torch
from botorch.utils.transforms import unnormalize


class Optimization():
    def __init__(self,
                 batch_size,
                 num_restarts_opti,
                 device,
                 dtype,
                 raw_samples):

        self.batch_size=batch_size
        self.num_restarts_opti=num_restarts_opti
        self.device=device
        self.dtype=dtype
        self.raw_samples=raw_samples

    def optimize_acqf_and_get_observation(self,
                                        acq_func,
                                        bounds):
        

        dim = acq_func.model.train_inputs[0].size(dim=1)

        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=torch.tensor([[0.0] * dim, [1.0] * dim], device=self.device, dtype=self.dtype),
            q=self.batch_size,
            num_restarts=self.num_restarts_opti,
            raw_samples=self.raw_samples,  
            options={"batch_limit": 5, "maxiter": 200}
        )

        new_x = candidates.detach()

        unnormalized_new_x=unnormalize(new_x, bounds)
        
        return unnormalized_new_x