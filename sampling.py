import torch


def sample_images(args, model, initial_x=None, num_samples=16):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        if initial_x is not None:
            x = initial_x.to(device)
        else:
            x = torch.randn((num_samples, 1, 28, 28)).to(device)
        for t in reversed(range(args.noise_step)):
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

def sample_z_given_x_y(y, x, rho):
    # Compute mean and variance for P(z|y,x)
    mean_numerator = y + x / (2 * rho**2)
    mean_denominator = 1 + 1/(2 * rho**2)
    mean = mean_numerator / mean_denominator
    variance = 1 / (2 * mean_denominator)
    
    # Sample z
    noise = torch.randn_like(mean) * torch.sqrt(torch.tensor(variance))
    z = mean + noise
    return z

def gibbs_sampling(args, y, model, num_iterations=100):
    # Initialize x with noise
    x = torch.randn_like(y)
    for _ in range(num_iterations):
        # Step 1: Sample z given x and y
        z = sample_z_given_x_y(y, x, args.rho)
        # Step 2: Sample x given z using diffusion model
        x = sample_images(args, model, initial_x=z, num_samples=z.shape[0])
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