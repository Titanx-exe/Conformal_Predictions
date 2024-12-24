import torch

import numpy
'''
mu, sigma = 0, 1
ds=numpy.random.normal(mu,sigma,100)
print(ds)
'''
prev_noise=None

def add_gausian_noise(optimizer,device,mu=0.0,sigma=1.0):
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad:
                #num_parameters=p.numel()
                #noise=torch.tensor(numpy.random.normal(mu,sigma,num_parameters),dtype=float,device=device)
                noise = torch.normal(mu, sigma, size=p.size(), device=device)
                #noise=noise.view(p.size())
                p.data.add_(optimizer.defaults["lr"],noise)
'''
def remove_noise(optimizer,noise_tensors):
    for gind in range(len(optimizer.param_groups)):
        noise_group=noise_tensors[gind]
        group=optimizer.param_groups[gind]["params"]
        for pind in range(len(group)):
            p=group[pind]
            noise=noise_group[pind]
            noise.mul_(-1)
            p.data.add_(noise)
'''


def add_anticorrelated_noise_prev_term(optimizer, device, alpha_0=0.1, sigma=1.0):
    """Applies anticorrelated noise based on previous noise term with dynamic alpha."""
    global prev_noise
    if prev_noise is None:
        # Initialize the previous noise as zero for each parameter
        prev_noise = {id(p): torch.zeros_like(p, device=device) for group in optimizer.param_groups for p in
                      group['params'] if p.requires_grad}

    for group in optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad:
                # Calculate dynamic alpha based on the current gradient norm
                grad_norm = p.grad.norm() + 1e-8  # Avoid division by zero
                alpha = alpha_0 / (1 + grad_norm)
                # Get the previous noise for this parameter
                noise_prev = prev_noise[id(p)]
                # Calculate new noise
                noise_new = -alpha * noise_prev + torch.normal(0, sigma, size=p.size(), device=device)
                # Apply noise to parameter and update stored noise
                p.data.add_(optimizer.defaults["lr"], noise_new)
                prev_noise[id(p)] = noise_new  # Store new noise term as previous noise for next iteration


'''    
prev_noise = None
# Within each training step, call this function and update `prev_noise`
prev_noise = add_anticorrelated_noise_method2_dynamic_alpha(optimizer, device, alpha_0=0.1, sigma=1.0, prev_noise=prev_noise)
'''


def add_anticorrelated_noise_gradient(optimizer, device, beta_0=50, sigma=70):
    """Applies gradient-based anticorrelated noise with dynamic beta."""

    for group in optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad:
                # Calculate dynamic beta based on the current gradient norm
                grad_norm = p.grad.norm() + 1e-8  # Add small epsilon to avoid division by zero
                beta = beta_0 / (1 + grad_norm)
                # Generate anticorrelated noise
                noise = -beta * p.grad + torch.normal(0, sigma, size=p.size(), device=device)
                # Apply noise to parameter
                p.data.add_(optimizer.defaults["lr"], noise)



