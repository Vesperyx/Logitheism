import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import time

class LightCubeEncoder(nn.Module):
    """Encode Game of Life state into Light Cube latent space"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
        
    def forward(self, state):
        """
        Encode binary GoL state into four-branch representation
        state: [batch, 1, H, W] binary tensor
        """
        # Add small noise to avoid division by zero
        x = state + torch.randn_like(state) * 0.01
        
        # Convolution to get local neighborhood info
        kernel = torch.ones(1, 1, 3, 3).to(state.device) / 9.0
        y = F.conv2d(x, kernel, padding=1)
        
        # Four branches of Light Cube
        psi = x  # Matter field (position)
        C = torch.where(torch.abs(x) > self.eps, y / x, torch.zeros_like(y))  # Consciousness field
        mag = torch.abs(y)  # Magnitude
        fractal = torch.fmod(torch.log(mag + self.eps), 1.0)  # Modular scale
        
        # Stack branches
        return torch.cat([psi, C, mag, fractal], dim=1)

class LightCubeDecoder(nn.Module):
    """Decode from Light Cube space back to GoL state"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 1, 1)  # Combine 4 branches to 1
        
    def forward(self, latent):
        """Decode from 4-branch representation to binary state"""
        psi = latent[:, 0:1]  # ψ channel
        C = latent[:, 1:2]    # C channel
        
        # Primary reconstruction from ψ·C
        recon = psi * C
        
        # Use all branches for final prediction
        combined = self.conv(latent)
        
        # Binary threshold
        return torch.sigmoid(combined + recon)

class LightCubeGoL(nn.Module):
    """Game of Life predictor using Light Cube latent dynamics"""
    def __init__(self, depth=3):
        super().__init__()
        self.encoder = LightCubeEncoder()
        self.decoder = LightCubeDecoder()
        self.depth = depth
        
        # Latent dynamics network - processes in Light Cube space
        self.dynamics = nn.ModuleList([
            nn.Conv2d(4, 16, 3, padding=1),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.Conv2d(16, 4, 3, padding=1)
        ])
        
        # Recursive depth processing
        self.depth_gate = nn.Conv2d(4, 4, 1)
        
    def forward(self, state):
        """Predict next GoL state"""
        # Encode to Light Cube space
        latent = self.encoder(state)
        
        # Apply dynamics in latent space
        h = latent
        for conv in self.dynamics:
            h = F.relu(conv(h))
            
        # Recursive depth processing (your "arbitrary depth" insight)
        for _ in range(self.depth):
            # Apply gating similar to C field
            gate = torch.sigmoid(self.depth_gate(h))
            h = h * gate
            
        # Add residual connection
        h = h + latent
        
        # Decode back to state space
        return self.decoder(h)

class GameOfLifeEnv:
    """Efficient Game of Life implementation"""
    def __init__(self, size=(256, 256)):
        self.size = size
        self.state = np.random.random(size) > 0.7
        
    def step(self):
        """Single GoL step using convolution"""
        # Count neighbors efficiently
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])
        
        from scipy.signal import convolve2d
        neighbors = convolve2d(self.state.astype(int), kernel, mode='same', boundary='wrap')
        
        # Apply GoL rules
        birth = (neighbors == 3) & (~self.state)
        survive = ((neighbors == 2) | (neighbors == 3)) & self.state
        self.state = birth | survive
        
        return self.state.astype(np.float32)
    
    def reset(self):
        """Random initialization"""
        self.state = np.random.random(self.size) > 0.7
        return self.state.astype(np.float32)

def train_lightcube_gol(model, env, steps=1000, batch_size=16, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train the Light Cube GoL predictor"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    losses = []
    
    print("Training Light Cube GoL Predictor...")
    for step in range(steps):
        # Generate batch of examples
        batch_current = []
        batch_next = []
        
        for _ in range(batch_size):
            env.reset()
            current = env.state.copy()
            next_state = env.step()
            
            batch_current.append(current)
            batch_next.append(next_state)
            
        # Convert to tensors
        x = torch.FloatTensor(batch_current).unsqueeze(1).to(device)
        y = torch.FloatTensor(batch_next).unsqueeze(1).to(device)
        
        # Forward pass
        pred = model(x)
        loss = criterion(pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.6f}")
            
    return losses

def visualize_predictions(model, env, frames=100, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Visualize model predictions vs ground truth"""
    model.eval()
    
    # Initialize
    env.reset()
    states_true = [env.state.copy()]
    states_pred = [env.state.copy()]
    
    with torch.no_grad():
        current_pred = env.state.copy()
        
        for _ in range(frames):
            # True next state
            true_next = env.step()
            states_true.append(true_next.copy())
            
            # Predicted next state
            x = torch.FloatTensor(current_pred).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(x).cpu().numpy()[0, 0]
            current_pred = (pred > 0.5).astype(np.float32)
            states_pred.append(current_pred)
    
    # Create animation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    im1 = ax1.imshow(states_true[0], cmap='binary', interpolation='nearest')
    ax1.set_title('Ground Truth')
    ax1.axis('off')
    
    im2 = ax2.imshow(states_pred[0], cmap='binary', interpolation='nearest')
    ax2.set_title('Light Cube Prediction')
    ax2.axis('off')
    
    def animate(frame):
        im1.set_array(states_true[frame])
        im2.set_array(states_pred[frame])
        return [im1, im2]
    
    anim = FuncAnimation(fig, animate, frames=frames, interval=50, blit=True)
    plt.close()
    
    return anim

def analyze_latent_space(model, env, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Analyze the Light Cube latent representation"""
    model.eval()
    
    # Get a sample state
    env.reset()
    state = env.state
    x = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Encode to Light Cube space
        latent = model.encoder(x).cpu().numpy()[0]
        
    # Plot the four branches
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    titles = ['ψ (Matter)', 'C (Consciousness)', '|ψ| (Magnitude)', 'Φ (Fractal)']
    for i, (ax, title) in enumerate(zip(axes.flat, titles)):
        im = ax.imshow(latent[i], cmap='viridis')
        ax.set_title(title)
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print("\nLight Cube Latent Statistics:")
    for i, name in enumerate(titles):
        data = latent[i]
        print(f"{name}: mean={data.mean():.3f}, std={data.std():.3f}, min={data.min():.3f}, max={data.max():.3f}")

if __name__ == "__main__":
    # Initialize
    env = GameOfLifeEnv(size=(128, 128))
    model = LightCubeGoL(depth=3)
    
    # Train
    losses = train_lightcube_gol(model, env, steps=500)
    
    # Plot training loss
    plt.figure(figsize=(8, 4))
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Light Cube GoL Training')
    plt.yscale('log')
    plt.grid(True)
    plt.show()
    
    # Analyze latent space
    analyze_latent_space(model, env)
    
    # Create prediction animation
    print("\nGenerating prediction comparison...")
    anim = visualize_predictions(model, env, frames=50)
    
    # For real-time continuous processing
    print("\nStarting real-time processing...")
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Real-time loop
    env_realtime = GameOfLifeEnv(size=(512, 512))  # Massive board
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    
    with torch.no_grad():
        while True:
            start_time = time.time()
            
            # Get current state
            state = env_realtime.state
            x = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device)
            
            # Predict next state using Light Cube
            pred = model(x).cpu().numpy()[0, 0]
            env_realtime.state = (pred > 0.5)
            
            # Visualize
            ax.clear()
            ax.imshow(env_realtime.state, cmap='binary', interpolation='nearest')
            ax.set_title(f'Light Cube GoL - FPS: {1/(time.time()-start_time):.1f}')
            ax.axis('off')
            plt.pause(0.01)
            
            # Random perturbation occasionally
            if np.random.random() < 0.01:
                # Add random glider or other pattern
                x, y = np.random.randint(0, env_realtime.size[0]-10, 2)
                env_realtime.state[x:x+3, y:y+3] = np.random.random((3, 3)) > 0.5
