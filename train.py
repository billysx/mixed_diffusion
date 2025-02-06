import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
import argparse

def langevin_monte_carlo(mu, sigma, num_samples, step_size, burn_in):
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

    for i in range(num_samples + burn_in):
        # Compute the gradient of the log probability
        grad = -np.linalg.inv(sigma) @ (x - mu)

        # Langevin dynamics update
        x = x + 0.5 * step_size * grad + np.sqrt(step_size) * np.random.randn(d)

        # Store the sample after burn-in
        if i >= burn_in:
            samples.append(x)

    return np.array(samples)

def train_diffusion_model(data):
    """
    Train a diffusion model to estimate the score function.
    Args:
        data (np.ndarray): Training data for the diffusion model.

    Returns:
        callable: Trained diffusion model that estimates the score.
    """
    # Use a simple neural network as the diffusion model
    model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=500, random_state=42)

    # Prepare training data for the score function
    x_train = data
    y_train = -data  # Score is proportional to -data for simplicity in this example

    model.fit(x_train, y_train)

    def diffusion_model(x):
        return model.predict(x)

    return diffusion_model

def estimated_score(x, model):
    """
    Estimate the score function using the diffusion model.
    Args:
        x (np.ndarray): Current data point.
        model (callable): Pre-trained diffusion model that estimates the score.

    Returns:
        np.ndarray: Estimated score.
    """
    return model(x)

def sample_from_diffusion(y, noise_std, num_iterations, langevin_step_size, langevin_burn_in, diffusion_model):
    """
    Sample from the actual distribution using a diffusion model and plug-and-play method.
    Args:
        y (np.ndarray): Noisy data points.
        noise_std (float): Standard deviation of the Gaussian noise.
        num_iterations (int): Number of iterations to run the algorithm.
        langevin_step_size (float): Step size for Langevin Monte Carlo.
        langevin_burn_in (int): Burn-in period for Langevin Monte Carlo.
        diffusion_model (callable): Pre-trained diffusion model that estimates the score function.

    Returns:
        np.ndarray: Generated samples approximating the actual data distribution.
    """
    x_t = np.random.randn(*y.shape)  # Initialize x_t randomly
    samples = []

    for iteration in range(num_iterations):
        # Step 1: Sample from the conditional distribution using Langevin Monte Carlo
        conditional_mean = (y + x_t) / 2
        # Change: Use y.shape[1] (data dimensionality) for conditional_cov
        conditional_cov = np.eye(y.shape[1]) * (noise_std ** 2) / 2

        sampled_x = langevin_monte_carlo(conditional_mean[iteration % y.shape[0]], conditional_cov, 1, langevin_step_size, langevin_burn_in)[0]

        # Step 2: Update x_t using the sampled_x and estimated score function
        score = estimated_score(x_t[iteration % y.shape[0]].reshape(1, -1), diffusion_model)
        x_t[iteration % y.shape[0]] = sampled_x + score.squeeze() * langevin_step_size

        # Store the current sample
        samples.append(x_t)

    return np.array(samples)

# Example setup
np.random.seed(42)
data_dim = 2
true_data = np.random.randn(100, data_dim)  # Generate some true data points
y = true_data + np.random.normal(scale=0.5, size=true_data.shape)  # Add Gaussian noise

# Train a diffusion model on true data
diffusion_model = train_diffusion_model(true_data)

# Parameters for the algorithm
num_iterations = 1000
langevin_step_size = 0.01
langevin_burn_in = 10
noise_std = 0.5

# Run the sampling algorithm
samples = sample_from_diffusion(y, noise_std, num_iterations, langevin_step_size, langevin_burn_in, diffusion_model)

# Visualize the results
plt.figure(figsize=(8, 6))
plt.scatter(y[:, 0], y[:, 1], alpha=0.5, label="Noisy Data (y)")
plt.scatter(true_data[:, 0], true_data[:, 1], alpha=0.5, label="True Data (x)")
plt.scatter(samples[-1][:, 0], samples[-1][:, 1], alpha=0.5, label="Generated Samples")
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Sampling from Actual Distribution using Diffusion Model")
plt.grid(alpha=0.3)
plt.show()




# add arguments
parser = argparse.ArgumentParser(description='Train a diffusion model')
parser.add_argument('--epochs', type=int, default=16, help='Number of epochs to train the model')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
parser.add_argument('--beta_start', type=float, default=0.0001, help='Initial noise level')
parser.add_argument('--beta_end', type=float, default=0.02, help='Final noise level')
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to train the model on')



args = parser.parse_args()