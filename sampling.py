import torch
from torch import nn, optim
from tqdm import tqdm
import numpy as np



def log_prob(x, mu, sigma):
    """
    Log of the target probability distribution (Gaussian in this example).
    Args:
        x (np.ndarray): Point at which to evaluate the log probability.
        mu (np.ndarray): Mean of the Gaussian.
        sigma (np.ndarray): Covariance matrix of the Gaussian.

    Returns:
        float: Log probability of x.
    """
    d = len(mu)
    diff = x - mu
    return -0.5 * (np.log((2 * np.pi)**d * np.linalg.det(sigma)) + diff.T @ np.linalg.inv(sigma) @ diff)

def gradient_log_prob(x, mu, sigma):
    """
    Gradient of the log probability with respect to x.
    Args:
        x (np.ndarray): Point at which to evaluate the gradient.
        mu (np.ndarray): Mean of the Gaussian.
        sigma (np.ndarray): Covariance matrix of the Gaussian.

    Returns:
        np.ndarray: Gradient of the log probability at x.
    """
    return -np.linalg.inv(sigma) @ (x - mu)

def langevin_monte_carlo(args, mu, sigma):
    """
    Langevin Monte Carlo sampling.
    Args:
        mu (np.ndarray): Mean of the target Gaussian distribution.
        sigma (np.ndarray): Covariance matrix of the target Gaussian.
        num_samples (int): Number of samples to generate.
        step_size (float): Step size for the Langevin dynamics.
        burn_in (int): Number of iterations to discard as burn-in.

    Returns:
        np.ndarray: Generated samples.
    """
    mu = mu.detach().cpu().numpy()
    sigma = sigma.detach().cpu().numpy()

    d = mu.shape[0]
    samples = []

    # Initialize the chain at a random point
    x = np.random.randn(d)

    for i in range(args.num_samples + args.burn_in):
        # Compute the gradient of the log probability
        grad = gradient_log_prob(x, mu, sigma)

        # Langevin dynamics update
        x = x + 0.5 * args.step_size * grad + np.sqrt(args.step_size) * np.random.randn(d)

        # Store the sample after burn-in
        if i >= args.burn_in:
            samples.append(x)

    return np.array(samples)

def naive_sampling(args, mu, sigma):
    """
    Naive sampling from a Gaussian distribution.
    Args:
        mu (np.ndarray): Mean of the target Gaussian distribution.
        sigma (np.ndarray): Covariance matrix of the target Gaussian.
        num_samples (int): Number of samples to generate.

    Returns:
        np.ndarray: Generated samples.
    """
    # if mu is a 2d tensor, like (1, 28, 28), then mu = mu.reshape(-1)
    res = np.random.multivariate_normal(mu, sigma, args.num_samples)
    return res


def sample_images(args, model, initial_x=None, num_samples=16):
    alphas = 1 - torch.linspace(args.beta_start, args.beta_end, args.noise_step)
    betas = 1 - alphas
    alphas_cumprod = torch.cumprod(alphas, 0)

    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        if initial_x is not None:
            x = initial_x.to(device)
        else:
            x = torch.randn((num_samples, 1, 28, 28)).to(device)
        for t in tqdm(reversed(range(args.noise_step))):
            t_tensor = torch.tensor([t] * x.shape[0]).to(device)
            noise_pred = model(x, t_tensor)
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]

            # Update x_t to x_{t-1}
            x = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * noise_pred)
            if t > 0:
                noise = torch.randn_like(x)
                x += torch.sqrt(betas[t]) * noise

        x = x.clamp(-1, 1)
    return x

def sample_z_given_x_y(args, y, x, rho):
    # mean and variance of P(z|y,x)
    mean_numerator = y + x / (2 * rho**2)
    mean_denominator = 1 + 1/(2 * rho**2)
    mean = mean_numerator / mean_denominator
    # variance = 1 / (2 * mean_denominator)
    d = mean.shape[0]
    variance = torch.eye(d) * (1 / (2 * mean_denominator))
    
    # Sample z from P(z|y,x)
    # z = naive_sampling(args, mean, variance)

    z = langevin_monte_carlo(args, mean, variance)
    
    return z

def gibbs_sampling(args, y, model, num_iterations=10):
    # Step 1: Sample z given x and y
    # Step 2: Sample x given z using diffusion model
    shape = y.shape
    y = y.flatten()
    x = torch.randn_like(y)
    for _ in tqdm(range(num_iterations)):
        z = sample_z_given_x_y(args, y, x, args.rho)
        x = sample_images(args, model, initial_x=z, num_samples=z.shape[0])
    x = x.reshape(shape)
    return x









# def sample_images(model, num_samples=16, ):
#     model.eval()
#     device = next(model.parameters()).device
#     with torch.no_grad():
#         x = torch.randn((num_samples, 1, 28, 28)).to(device)  # Start with pure noise
#         for t in reversed(range(args.noise_step)):
#             t_tensor = torch.tensor([t] * num_samples).to(device)
#             noise_pred = model(x, t_tensor)
#             alpha_t = alphas[t]
#             alpha_cumprod_t = alphas_cumprod[t]

#             # Update x_t to x_t-1
#             x = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * noise_pred)
#             if t > 0:
#                 noise = torch.randn_like(x)
#                 x += torch.sqrt(betas[t]) * noise

#         x = x.clamp(-1, 1)
#         return x



# def sample_z_given_x_y(y, x, epsilon=0.1):
#     return y + epsilon * np.random.randn(*x.shape)

# def sample_x_given_z(z, diffusion_model, epsilon=0.1):
#     z_tensor = torch.tensor(z, dtype=torch.float32)
#     with torch.no_grad():
#         predicted_noise = diffusion_model(z_tensor).numpy()
#     return z - epsilon * predicted_noise

# def gibbs_sampling(y, diffusion_model, num_samples=100, epsilon=0.1):
#     x_samples = np.random.randn(*y.shape)  # Initialize x
#     for _ in range(num_samples):
#         z_samples = sample_z_given_x_y(y, x_samples, epsilon)
#         x_samples = sample_x_given_z(z_samples, diffusion_model, epsilon)
#     return x_samples