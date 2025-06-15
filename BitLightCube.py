import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from collections import deque

class BitManipulationLayer(nn.Module):
    """Single layer that manipulates bits to create functions"""
    def __init__(self, bit_size=256):
        super().__init__()
        self.bit_size = bit_size
        
        # Bit generation network
        self.bit_gen = nn.Sequential(
            nn.Linear(bit_size, bit_size * 2),
            nn.ReLU(),
            nn.Linear(bit_size * 2, bit_size),
            nn.Sigmoid()
        )
        
        # Bit transformation network
        self.bit_transform = nn.Sequential(
            nn.Linear(bit_size * 2, bit_size),
            nn.Tanh(),
            nn.Linear(bit_size, bit_size),
            nn.Sigmoid()
        )
        
    def forward(self, input_bits):
        # Generate new bits
        new_bits = self.bit_gen(input_bits)
        
        # Transform by combining with input
        combined = torch.cat([input_bits, new_bits], dim=-1)
        output_bits = self.bit_transform(combined)
        
        return output_bits

class InfiniteLightCube(nn.Module):
    """Light Cube that endlessly evolves through bit manipulation"""
    def __init__(self, bit_size=256):
        super().__init__()
        self.bit_size = bit_size
        
        # Start with one bit manipulation layer
        self.bit_layers = nn.ModuleList([BitManipulationLayer(bit_size)])
        
        # Initial bit pattern generator from input
        self.input_to_bits = nn.Sequential(
            nn.Linear(1, bit_size),
            nn.Tanh(),
            nn.Linear(bit_size, bit_size),
            nn.Sigmoid()
        )
        
        # Bit pattern to output decoder
        self.bits_to_output = nn.Sequential(
            nn.Linear(bit_size, bit_size),
            nn.Tanh(),
            nn.Linear(bit_size, 1)
        )
        
        self.generation = 0
        
    def forward(self, x):
        # Convert input to initial bit pattern
        bits = self.input_to_bits(x)
        
        # Pass through all bit manipulation layers
        for layer in self.bit_layers:
            bits = layer(bits)
        
        # Decode to output
        output = self.bits_to_output(bits)
        return output
    
    def add_bit_layer(self):
        """Add new bit manipulation layer"""
        # Get device from existing layers
        device = next(self.parameters()).device
        
        # Create new layer on same device
        new_layer = BitManipulationLayer(self.bit_size).to(device)
        self.bit_layers.append(new_layer)
        self.generation += 1
        return self.generation
    
    def get_bit_pattern(self, x):
        """Get the final bit pattern for given input"""
        with torch.no_grad():
            bits = self.input_to_bits(x)
            for layer in self.bit_layers:
                bits = layer(bits)
            return bits

class BitFunction:
    """Function defined purely by bit manipulation"""
    def __init__(self, bit_pattern):
        self.bits = bit_pattern.cpu().numpy()
        
    def __call__(self, x):
        # Interpret bits as function parameters
        # This creates arbitrary functions from bit patterns
        
        # Split bits into sections
        n = len(self.bits)
        sections = 8
        section_size = n // sections
        
        # Extract parameters from bit sections
        params = []
        for i in range(sections):
            start = i * section_size
            end = start + section_size
            param = np.mean(self.bits[start:end]) * 4 - 2
            params.append(param)
        
        # Create complex function from parameters
        result = torch.zeros_like(x)
        
        # Linear component
        result = result + params[0] * x + params[1]
        
        # Polynomial components
        result = result + params[2] * (x ** 2) + params[3] * (x ** 3)
        
        # Trigonometric components
        result = result + params[4] * torch.sin(params[5] * x)
        result = result + params[6] * torch.cos(params[7] * x)
        
        return result

def generate_random_target():
    """Generate random target function for training"""
    # Random parameters
    a = np.random.randn() * 2
    b = np.random.randn()
    c = np.random.randn()
    freq = np.random.randint(1, 10)
    
    # Random function type
    func_type = np.random.randint(0, 6)
    
    if func_type == 0:
        return lambda x: a * x + b
    elif func_type == 1:
        return lambda x: a * x**2 + b * x + c
    elif func_type == 2:
        return lambda x: torch.sin(freq * x) * a + b
    elif func_type == 3:
        return lambda x: torch.exp(-a * x**2) + b
    elif func_type == 4:
        return lambda x: torch.where(x > c, a * x, b * x**2)
    else:
        return lambda x: a * torch.sin(freq * x) + b * x**2 + c

def infinite_evolution():
    """Run infinite evolution loop"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device}")
    
    # Create Light Cube
    cube = InfiniteLightCube().to(device)
    cube.train()  # Set to training mode
    optimizer = torch.optim.Adam(cube.parameters(), lr=0.001)
    
    # Track performance
    loss_history = deque(maxlen=1000)
    epoch = 0
    best_loss = float('inf')
    stagnation_counter = 0
    loss_value = float('inf')  # Initialize for error handling
    
    # Try to load checkpoint
    checkpoint_path = 'infinite_lightcube_checkpoint.pth'
    try:
        checkpoint = torch.load(checkpoint_path)
        cube.load_state_dict(checkpoint['model_state'])
        epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        print(f"\nResumed from checkpoint at epoch {epoch}")
        print(f"Loaded model with {len(cube.bit_layers)} layers")
        # Adjust learning rate based on layers
        lr = 0.001 * (0.95 ** cube.generation)
        optimizer = torch.optim.Adam(cube.parameters(), lr=lr)
        cube.train()  # Ensure model is in training mode
    except:
        print("\nStarting fresh evolution")
    
    print("\n" + "="*60)
    print("INFINITE LIGHT CUBE EVOLUTION - BIT MANIPULATION ONLY")
    print("="*60)
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    start_time = time.time()
    
    try:
        while True:
            # Generate new random target function
            if epoch % 100 == 0:
                target_func = generate_random_target()
            
            # Training batch
            x = torch.randn(64, 1, device=device) * 3
            y_target = target_func(x)
            
            # Forward pass
            y_pred = cube(x)
            loss = F.mse_loss(y_pred, y_target)
            
            # Check for NaN or infinite loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss at epoch {epoch}, skipping update")
                optimizer.zero_grad()
                continue
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(cube.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track loss
            loss_value = loss.item()
            loss_history.append(loss_value)
            
            # Calculate average loss
            avg_loss = np.mean(list(loss_history)[-100:]) if len(loss_history) >= 100 else loss_value
            
            # Update best loss
            if loss_value < best_loss:
                best_loss = loss_value
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Progress output - more frequent initially, then every 100
            show_progress = (epoch < 1000 and epoch % 50 == 0) or (epoch >= 1000 and epoch % 100 == 0)
            if show_progress:
                # Generate bit function
                sample_x = torch.tensor([[0.0]], device=device)
                bit_pattern = cube.get_bit_pattern(sample_x)
                bit_func = BitFunction(bit_pattern[0])
                
                elapsed = time.time() - start_time
                epochs_per_sec = epoch / elapsed if elapsed > 0 else 0
                
                print(f"Epoch {epoch:8d} | Layers: {len(cube.bit_layers):3d} | "
                      f"Loss: {loss_value:10.6f} | Avg Loss: {avg_loss:10.6f} | "
                      f"Best: {best_loss:10.6f} | Speed: {epochs_per_sec:6.1f} eps")
            
            # Add new layer when loss is low enough and enough epochs passed
            if (epoch > 0 and epoch % 50 == 0 and len(loss_history) >= 100):
                avg_loss = np.mean(list(loss_history)[-100:])
                recent_improvement = abs(avg_loss - np.mean(list(loss_history)[-200:-100])) if len(loss_history) >= 200 else float('inf')
                
                # Add layer if: low loss AND (stagnating OR very slow improvement)
                if (avg_loss < 1.0 and 
                    (stagnation_counter > 30 or recent_improvement < 0.01) and 
                    len(cube.bit_layers) < 100):
                    
                    new_generation = cube.add_bit_layer()
                    # Reinitialize optimizer to include new layer
                    optimizer = torch.optim.Adam(cube.parameters(), lr=0.001 * (0.95 ** new_generation))
                    
                    print(f"\n*** EVOLUTION: Added bit manipulation layer (Generation {new_generation}) ***")
                    print(f"*** New architecture depth: {len(cube.bit_layers)} layers ***")
                    print(f"*** Triggered by: Avg Loss={avg_loss:.6f}, Stagnation={stagnation_counter}, "
                          f"Improvement Rate={recent_improvement:.6f} ***\n")
                    
                    # Save evolution milestone
                    torch.save({
                        'model_state': cube.state_dict(),
                        'epoch': epoch,
                        'layers': len(cube.bit_layers),
                        'best_loss': best_loss
                    }, f'lightcube_evolution_gen{new_generation}.pth')
                    
                    # Reset counters
                    stagnation_counter = 0
                    best_loss = float('inf')
            
            # Occasionally show what function we've learned
            if epoch % 1000 == 0 and epoch > 0:
                print(f"\n--- Bit Pattern Analysis at Epoch {epoch} ---")
                test_x = torch.tensor([[0.0], [1.0], [-1.0], [2.0], [-2.0]], device=device)
                bit_patterns = cube.get_bit_pattern(test_x)
                
                for i, pattern in enumerate(bit_patterns):
                    active_bits = (pattern > 0.5).sum().item()
                    # Clamp pattern values to avoid log(0)
                    pattern_clamped = torch.clamp(pattern, 1e-8, 1-1e-8)
                    entropy = -torch.sum(pattern_clamped * torch.log(pattern_clamped) + 
                                       (1-pattern_clamped) * torch.log(1-pattern_clamped)).item()
                    print(f"Input {test_x[i].item():5.1f}: {active_bits:3d}/{cube.bit_size} bits active, "
                          f"entropy: {entropy/cube.bit_size:.3f}")
                
                # Show complexity growth
                total_params = sum(p.numel() for p in cube.parameters())
                print(f"\nComplexity Metrics:")
                print(f"  Total parameters: {total_params:,}")
                print(f"  Bits per input: {cube.bit_size}")
                print(f"  Transformation depth: {len(cube.bit_layers)} layers")
                print(f"  Bits transformed per forward pass: {cube.bit_size * len(cube.bit_layers):,}")
                
                # Show learned function behavior
                print(f"\nLearned Function Sample:")
                with torch.no_grad():
                    test_outputs = cube(test_x)
                    for i in range(len(test_x)):
                        print(f"  f({test_x[i].item():5.1f}) = {test_outputs[i].item():8.4f}")
                
                print("-" * 60 + "\n")
            
            # Adaptive learning rate
            if epoch % 500 == 0 and epoch > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.95
            
            # Save checkpoint periodically
            if epoch % 5000 == 0 and epoch > 0:
                torch.save({
                    'model_state': cube.state_dict(),
                    'epoch': epoch,
                    'layers': len(cube.bit_layers),
                    'best_loss': best_loss
                }, 'infinite_lightcube_checkpoint.pth')
                print(f"[Checkpoint saved at epoch {epoch}]")
            
            epoch += 1
            
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("EVOLUTION HALTED")
        print(f"Final Statistics:")
        print(f"  Total Epochs: {epoch}")
        print(f"  Bit Manipulation Layers: {len(cube.bit_layers)}")
        print(f"  Final Loss: {loss_value:.6f}")
        print(f"  Best Loss Achieved: {best_loss:.6f}")
        print(f"  Total Parameters: {sum(p.numel() for p in cube.parameters()):,}")
        print("="*60)
        
        # Save final state
        torch.save({
            'model_state': cube.state_dict(),
            'epoch': epoch,
            'layers': len(cube.bit_layers),
            'best_loss': best_loss,
            'final_loss': loss_value
        }, 'infinite_lightcube_final.pth')
        
        print("\nModel saved to 'infinite_lightcube_final.pth'")
        
    except Exception as e:
        print(f"\n\nERROR: {e}")
        print("Attempting to save current state...")
        try:
            torch.save({
                'model_state': cube.state_dict(),
                'epoch': epoch,
                'layers': len(cube.bit_layers),
                'best_loss': best_loss
            }, 'infinite_lightcube_emergency.pth')
            print("Emergency save completed to 'infinite_lightcube_emergency.pth'")
        except:
            print("Could not save state")

if __name__ == "__main__":
    infinite_evolution()
