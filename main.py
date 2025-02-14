import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from models import SimpleUNet
from tqdm import tqdm
from sampling import gibbs_sampling, sample_z_given_x_y


def add_noise(x0, t, alphas_cumprod):
    noise = torch.randn_like(x0)
    sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod[t]).to(x0.device).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - alphas_cumprod[t]).to(x0.device).view(-1, 1, 1, 1)
    return sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise, noise





def train(args, dataloader):
    # input_dim = 28 * 28 

    # diffusion_model = DiffusionModel(input_dim=input_dim)
    # optimizer = optim.Adam(diffusion_model.parameters(), lr=args.learning_rate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps"

    betas = torch.linspace(args.beta_start, args.beta_end, args.noise_step)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, 0).to(device)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).to(device), alphas_cumprod[:-1]])


    model = SimpleUNet(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    model.train()
    mse = nn.MSELoss()
    
    for epoch in range(args.epochs):
        pbar = tqdm(dataloader)
        total_loss = 0
        for i, (x0, _) in enumerate(pbar):
            x0 = x0.to(device)
            t = torch.randint(0, args.noise_step, (x0.shape[0],)).to(device)
            noisy_x, noise = add_noise(x0, t, alphas_cumprod)

            noise_pred = model(noisy_x, t)

            loss = mse(noise_pred, noise)
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        print(f"Epoch {epoch + 1} Loss: {total_loss/len(dataloader)}")
    return model



def main(args):


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_dataloader = DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(mnist_test, batch_size=1, shuffle=False)

    
    model = train(args, train_dataloader)
    
    x0, _ = next(iter(test_dataloader))
    x0 = x0.to(next(model.parameters()).device)
    noise = torch.randn_like(x0) * args.rho
    y = x0 + noise
    
    # Perform Gibbs sampling to denoise
    x_denoised = gibbs_sampling(args, y, model)
    print("Denoised samples shape:", x_denoised.shape)
    # Save the denoised samples
    torch.save(x_denoised, "denoised_samples.png")

# Example usage
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a diffusion model for image generation.")
    parser.add_argument("--noise_step", type=int, default=1000, help="Total diffusion steps.")
    parser.add_argument("--beta_start", type=float, default=0.0001, help="Initial noise level.")
    parser.add_argument("--beta_end", type=float, default=0.02, help="Final noise level.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=16, help="Number of epochs to train the model.")
    parser.add_argument("--data_file", type=str, default="./data", help="Path to the data directory.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training.")

    parser.add_argument("--num_samples", type=int, default=5000, help="Number of samples to generate for Langevin Monte Carlo.")
    parser.add_argument("--step_size", type=float, default=0.01, help="Step size for Langevin Monte Carlo.")
    parser.add_argument("--burn_in", type=int, default=1000, help="Number of burn-in iterations for Langevin Monte Carlo.")
    parser.add_argument("--rho", type=float, default=0.1, help="Noise level for generating noisy observation.")
    args = parser.parse_args()


    main(args)
