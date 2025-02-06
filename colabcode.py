import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import argparse
from models import SimpleUNet




def add_noise(x0, t):
    noise = torch.randn_like(x0)
    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t]).to(x0.device).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[t]).to(x0.device).view(-1, 1, 1, 1)
    return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise, noise





def train(args, dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    for epoch in range(args.epochs):
        for i, (x0, _) in enumerate(dataloader):
            x0 = x0.to(device)
            t = torch.randint(0, args.T, (x0.shape[0],)).to(device)  # Random timestep
            noisy_x, noise = add_noise(x0, t)

            # Predict the noise added
            noise_pred = model(noisy_x, t)

            # Loss based on predicting the noise
            loss = nn.MSELoss()(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1} Loss: {loss.item()}")



# generate new mnist images

def sample_images(model, num_samples=16):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.randn((num_samples, 1, 28, 28)).to(device)  # Start with pure noise
        for t in reversed(range(args.T)):
            t_tensor = torch.tensor([t] * num_samples).to(device)
            noise_pred = model(x, t_tensor)
            alpha_t = alphas[t]
            alpha_cumprod_t = alphas_cumprod[t]

            # Update x_t to x_t-1
            x = (1 / torch.sqrt(alpha_t)) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_cumprod_t) * noise_pred)
            if t > 0:
                noise = torch.randn_like(x)
                x += torch.sqrt(betas[t]) * noise

        x = x.clamp(-1, 1)
        return x





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
    d = len(mu)
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







def main(args):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root=args.data_file, train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "gpu")

    model = SimpleUNet().to(device)
    samples = sample_images(model)
    samples = samples.cpu().numpy()
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].reshape(28, 28), cmap="gray")
        ax.axis("off")
    plt.show()
    train(args, train_loader)

    # Define the parameters of the Gaussian distribution
    mu = np.array([0, 0])  # Mean
    sigma = np.array([[1, 0.5], [0.5, 1]])  # Covariance matrix


    samples = langevin_monte_carlo(args, mu, sigma)

    # Plot the generated samples
    plt.figure(figsize=(8, 6))
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5, label="Samples")
    plt.title("Langevin Monte Carlo Sampling")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(alpha=0.3)
    plt.axis("equal")
    plt.legend()
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a diffusion model for image generation.")
    parser.add_argument("--T", type=int, default=1000, help="Total diffusion steps.")
    parser.add_argument("--beta_start", type=float, default=0.0001, help="Initial noise level.")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Final noise level.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=16, help="Number of epochs to train the model.")
    parser.add_argument("--data_file", type=str, default="./data", help="Path to the data directory.")

    parser.add_argument("num_samples", type=int, default=5000, help="Number of samples to generate for Langevin Monte Carlo.")
    parser.add_argument("step_size", type=float, default=0.01, help="Step size for Langevin Monte Carlo.")
    parser.add_argument("burn_in", type=int, default=1000, help="Number of burn-in iterations for Langevin Monte Carlo.")
    args = parser.parse_args()


    betas = torch.linspace(args.beta_start, args.beta_end, args.T)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), alphas_cumprod[:-1]])
    print("Alphas:", alphas)
    print("Betas:", betas)
    print("Alphas Cumprod:", alphas_cumprod)
    print("Alphas Cumprod Prev:", alphas_cumprod_prev)



    main(args)


