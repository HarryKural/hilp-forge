"""
Diffusion Model for Tabular Data Generation
Implements DDPM (Denoising Diffusion Probabilistic Models) for tornado data
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


class TabularDataset(Dataset):
    """Dataset for tabular data"""
    
    def __init__(self, data):
        """
        Args:
            data: numpy array or pandas DataFrame of normalized features
        """
        if hasattr(data, 'values'):
            data = data.values
        self.data = torch.FloatTensor(data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class MLPDiffusionModel(nn.Module):
    """
    MLP-based diffusion model for tabular data generation;
    predicts noise added to data at each timestep during diffusion process;
    uses time embeddings to condition on diffusion timestep
    """
    
    def __init__(self, input_dim, hidden_dims=[512, 1024, 1024, 512], dropout=0.1):
        """
        Args:
            input_dim: Number of features
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # time embedding (sinusoidal encoding)
        self.time_embed_dim = 128
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embed_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Linear(hidden_dims[0], hidden_dims[0])
        )
        
        # main network
        layers = []
        prev_dim = input_dim + hidden_dims[0]  # concatenate with time embedding
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # output layer (predicts noise)
        layers.append(nn.Linear(prev_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x, t):
        """
        Args:
            x: Input data [batch_size, input_dim]
            t: Timestep [batch_size]
        Returns:
            Predicted noise [batch_size, input_dim]
        """
        # get time embeddings
        t_emb = self.get_timestep_embedding(t)
        t_emb = self.time_mlp(t_emb)
        
        # concatenate input and time embedding
        x = torch.cat([x, t_emb], dim=-1)
        
        # pass through network
        return self.network(x)
    
    def get_timestep_embedding(self, timesteps):
        """
        Sinusoidal timestep embeddings
        """
        half_dim = self.time_embed_dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class DiffusionTrainer:
    """
    Trainer for diffusion model
    Implements DDPM training and sampling
    """

    def __init__(self, model, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device=None):
        """
        Args:
            model: Neural network model
            num_timesteps: Number of diffusion steps
            beta_start: Starting beta value
            beta_end: Ending beta value
            device: 'cuda', 'mps', or 'cpu' (or None to auto-select)
        """
        import torch

        self.model = model
        self.num_timesteps = num_timesteps

        # determine a usable device string
        if device is None:
            if torch.cuda.is_available():
                chosen = 'cuda'
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                chosen = 'mps'
            else:
                chosen = 'cpu'
        else:
            # user requested device; verify availability
            req = str(device).lower()
            if req == 'cuda' and not torch.cuda.is_available():
                print("Requested device 'cuda' not available — falling back to 'cpu'.")
                chosen = 'cpu'
            elif req == 'mps' and not (getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()):
                print("Requested device 'mps' not available — falling back to 'cpu'.")
                chosen = 'cpu'
            else:
                chosen = req

        # convert to torch.device where possible
        try:
            self.device = torch.device(chosen)
        except Exception:
            # fallback to cpu
            self.device = torch.device('cpu')

        # move model to the selected device
        self.model.to(self.device)

        # beta schedule and other precomputed tensors should live on the trainer device
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # precompute helper tensors on device
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

        print(f"Diffusion trainer initialized on {self.device}")
        print(f"Number of timesteps: {num_timesteps}")
        print(f"Beta schedule: {beta_start} to {beta_end}")
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion: Add noise to data
        q(x_t | x_0)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, x_start, t, noise=None):
        """
        Compute loss for training
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # forward diffusion
        x_noisy = self.q_sample(x_start, t, noise)
        
        # predict noise
        predicted_noise = self.model(x_noisy, t)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss
    
    def train_epoch(self, dataloader, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            # sample random timesteps
            t = torch.randint(0, self.num_timesteps, (batch.shape[0],), device=self.device)
            
            # compute loss
            loss = self.p_losses(batch, t)
            
            # backprop
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def p_sample(self, x, t):
        """
        Sample from p(x_{t-1} | x_t)
        Reverse diffusion step
        """
        betas_t = self.betas[t][:, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t])[:, None]
        
        # predict noise
        predicted_noise = self.model(x, t)
        
        # Mean of p(x_{t-1} | x_t)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t][:, None]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, verbose=True):
        """
        Generate samples by iteratively denoising
        """
        self.model.eval()
        
        # start from pure noise
        x = torch.randn(shape, device=self.device)
        
        # iteratively denoise
        iterator = tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling', total=self.num_timesteps) if verbose else reversed(range(0, self.num_timesteps))
        
        for i in iterator:
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(x, t)
        
        return x
    
    @torch.no_grad()
    def sample(self, num_samples, verbose=True):
        """
        Generate new samples
        """
        shape = (num_samples, self.model.input_dim)
        samples = self.p_sample_loop(shape, verbose=verbose)
        return samples.cpu().numpy()


def train_diffusion_model(data, num_epochs=1000, batch_size=128, lr=1e-3,
                          hidden_dims=[256, 512, 256], num_timesteps=1000,
                          save_path='models/tornado_diffusion.pt'):
    """
    Complete training pipeline for diffusion model
    
    Args:
        data: Normalized training data (numpy array or DataFrame)
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        hidden_dims: Hidden layer dimensions
        num_timesteps: Number of diffusion timesteps
        save_path: Path to save trained model
    
    Returns:
        Trained model and trainer
    """
    print("="*60)
    print("TRAINING DIFFUSION MODEL")
    print("="*60)
    
    # prepare data
    dataset = TabularDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_dim = data.shape[1] if hasattr(data, 'shape') else data.values.shape[1]
    
    print(f"\nTraining Configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Training samples: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Hidden dimensions: {hidden_dims}")
    print(f"  Diffusion timesteps: {num_timesteps}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")
    
    # initialize model
    model = MLPDiffusionModel(input_dim=input_dim, hidden_dims=hidden_dims)
    trainer = DiffusionTrainer(model, num_timesteps=num_timesteps)
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    # training loop
    print("Starting training...")
    losses = []
    
    for epoch in range(num_epochs):
        loss = trainer.train_epoch(dataloader, optimizer)
        losses.append(loss)
        scheduler.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {loss:.6f}")
    
    print("\nTraining complete!")
    
    # save model
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'num_timesteps': num_timesteps,
        'losses': losses
    }, save_path)
    print(f"Model saved to: {save_path}")
    
    # plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Diffusion Model Training Loss', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    loss_plot_path = save_path.replace('.pt', '_loss.png')
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss plot saved to: {loss_plot_path}")
    
    return model, trainer


def load_trained_model(model_path, device=None):
    """
    Load a trained diffusion model

    Args:
        model_path: Path to saved model
        device: Desired device for trainer/model ('cuda', 'mps', 'cpu', or None for auto)
    Returns:
        Model and trainer
    """
    import torch

    # load checkpoint onto CPU first to avoid device-specific deserialization issues
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint '{model_path}' onto CPU: {e}")

    # reconstruct model on CPU
    model = MLPDiffusionModel(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['hidden_dims']
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    # create trainer — trainer will move model and tensors to the appropriate device
    trainer = DiffusionTrainer(
        model,
        num_timesteps=checkpoint.get('num_timesteps', 1000),
        device=device
    )

    print(f"Loaded model from: {model_path}")
    return model, trainer


if __name__ == "__main__":
    # example usage
    import pandas as pd
    
    # load normalized data
    data = pd.read_csv('data/processed/tornado_hilp_normalized.csv')
    
    # train model
    model, trainer = train_diffusion_model(
        data,
        num_epochs=1000,
        batch_size=64,
        lr=1e-3
    )
    
    # generate samples
    print("\nGenerating samples...")
    samples = trainer.sample(num_samples=100)
    print(f"Generated {len(samples)} synthetic samples")