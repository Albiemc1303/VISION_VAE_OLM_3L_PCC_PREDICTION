import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
from collections import deque
import math


class PatternLSTM(nn.Module):
    """Frozen LSTM for pattern detection in latent sequences"""
    
    def __init__(self, latent_dim: int, hidden_dim: int = 512, num_layers: int = 2, buffer_size: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.buffer_size = buffer_size
        
        # Flatten latent dimensions for LSTM input
        self.input_dim = latent_dim
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Initialize as frozen (no training)
        self.freeze_weights()
        
        # Rolling buffer for latent sequences
        self.latent_buffer = deque(maxlen=buffer_size)
        
    def freeze_weights(self):
        """Freeze all model parameters"""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def add_latent(self, latent: torch.Tensor):
        """Add latent to rolling buffer"""
        # Flatten spatial dimensions: [1, channels, height, width] -> [1, channels*height*width]
        flattened = latent.view(1, -1)
        self.latent_buffer.append(flattened)
    
    def detect_patterns(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect patterns in the latent sequence and return last output and full sequence outputs.

        Returns:
            last_output: [batch=1, seq=1, hidden_dim]
            full_output: [batch=1, seq_len, hidden_dim]
        """
        if len(self.latent_buffer) < 2:
            # Return zero tensors if insufficient data
            zero_last = torch.zeros(1, 1, self.hidden_dim, device=next(self.parameters()).device)
            zero_full = torch.zeros(1, 1, self.hidden_dim, device=next(self.parameters()).device)
            return zero_last, zero_full
        
        # Convert buffer to sequence tensor (current: [seq_len, 1, input_dim])
        sequence = torch.stack(list(self.latent_buffer))
        # Reorder to batch_first: [1, seq_len, input_dim]
        sequence = sequence.transpose(0, 1)
        
        with torch.no_grad():
            # LSTM forward (batch_first=True)
            lstm_output, (hidden_states, _) = self.lstm(sequence)  # [1, seq_len, hidden_dim]
            
            # Last time step output (keep seq dim = 1)
            last_output = lstm_output[:, -1:, :]
            
        return last_output, lstm_output


class CompressionLSTM(nn.Module):
    """LSTM that learns to compress patterns in real-time"""
    
    def __init__(self, pattern_dim: int, compressed_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.pattern_dim = pattern_dim
        self.compressed_dim = compressed_dim
        self.num_layers = num_layers
        
        # LSTM for compression
        self.lstm = nn.LSTM(
            input_size=pattern_dim,
            hidden_size=compressed_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Output projection
        self.output_projection = nn.Linear(compressed_dim, compressed_dim)
        self.activation = nn.Tanh()
        # Reconstruction head to map compressed features back to pattern dimension
        self.reconstruction_head = nn.Linear(compressed_dim, pattern_dim)
        
        # Optimizer for real-time training
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training state
        self.training_step = 0
        self.loss_history = deque(maxlen=100)
        
    def forward(self, patterns: torch.Tensor) -> torch.Tensor:
        """Compress pattern sequences.
        Accepts patterns in either [batch, seq_len, pattern_dim] (preferred) or [seq_len, batch, pattern_dim]."""
        if patterns.dim() != 3:
            raise ValueError("patterns must be 3D tensor")
        # Determine layout
        # If first dim is batch=1 and second dim > 1 and last dim ~ pattern_dim, likely [1, seq, H]
        # If first dim is seq_len and second dim is batch, we transpose.
        if patterns.size(0) != 1 and patterns.size(1) == 1:
            # [seq_len, batch, H] -> [batch, seq_len, H]
            patterns = patterns.transpose(0, 1)
        elif patterns.size(0) == 1 and patterns.size(1) >= 1:
            # Already [batch, seq_len, H]
            pass
        else:
            # Fallback assume [seq_len, batch, H]
            patterns = patterns.transpose(0, 1)

        batch_size = patterns.size(0)
        seq_len = patterns.size(1)
        pattern_dim = patterns.size(2)

        # Ensure pattern_dim matches expected input size
        if pattern_dim != self.pattern_dim:
            if pattern_dim > self.pattern_dim:
                patterns = patterns[:, :, :self.pattern_dim]
            else:
                padding = torch.zeros(batch_size, seq_len, self.pattern_dim - pattern_dim,
                                      device=patterns.device, dtype=patterns.dtype)
                patterns = torch.cat([patterns, padding], dim=2)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(patterns)
        
        # Use last hidden state
        last_hidden = hidden[-1]  # [batch, compressed_dim]
        
        # Project and activate
        compressed = self.output_projection(last_hidden)
        compressed = self.activation(compressed)
        
        return compressed
    
    def compress_patterns(self, patterns: torch.Tensor) -> torch.Tensor:
        """Compress patterns and optionally train.
        Accepts a short sequence [batch, seq_len, H] for temporal modeling."""
        do_train = (self.training_step % 5 == 0)
        if do_train:
            self.train()
            compressed = self.forward(patterns)
            self._train_step(patterns, compressed)
            self.eval()
        with torch.no_grad():
            compressed_out = self.forward(patterns)
        self.training_step += 1
        return compressed_out
    
    def _train_step(self, patterns: torch.Tensor, compressed: torch.Tensor):
        """Perform one training step with anti-collapse objective.
        Predict the next pattern vector from the sequence via compressed representation."""
        self.train()
        # Ensure patterns are [batch, seq_len, H]
        if patterns.dim() != 3:
            patterns = patterns.unsqueeze(0)
        if patterns.size(0) != 1 and patterns.size(1) == 1:
            patterns = patterns.transpose(0, 1)

        batch_size = patterns.size(0)
        seq_len = patterns.size(1)
        pattern_dim = patterns.size(2)

        # Use last two steps if available
        if seq_len >= 2:
            prev_pattern = patterns[:, -2, :]
            next_pattern = patterns[:, -1, :]
        else:
            prev_pattern = patterns[:, -1, :]
            next_pattern = patterns[:, -1, :]

        # Map compressed to pattern space to predict next
        predicted_next = self.reconstruction_head(compressed)  # [batch, H]

        # Loss: prediction MSE + small variance encouragement
        loss_pred = nn.MSELoss()(predicted_next, next_pattern)
        var = torch.var(predicted_next, dim=1).mean()
        loss_var = -0.001 * var
        loss = loss_pred + loss_var
        
        # Regularization
        l2_reg = sum(p.pow(2.0).sum() for p in self.parameters())
        loss += 1e-5 * l2_reg
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        self.scheduler.step(loss.item())
        
        # Track loss
        self.loss_history.append(loss.item())
        
        self.eval()
    
    def _reconstruct_from_compressed(self, compressed: torch.Tensor) -> torch.Tensor:
        """Simple reconstruction from compressed representation"""
        # This is a simplified reconstruction - could be made more sophisticated
        batch_size = compressed.size(0)
        
        # Create a basic reconstruction by expanding the compressed representation
        expanded = compressed.unsqueeze(1).repeat(1, self.num_layers, 1)
        
        # Project back to pattern dimension
        reconstructed = torch.zeros(batch_size, self.num_layers, self.pattern_dim, 
                                  device=compressed.device)
        
        # Simple linear projection for each layer
        for i in range(self.num_layers):
            layer_proj = nn.Linear(self.compressed_dim, self.pattern_dim, device=compressed.device)
            reconstructed[:, i] = layer_proj(expanded[:, i])
        
        return reconstructed


class CentralLSTM(nn.Module):
    """Central LSTM for next frame prediction with real-time training"""
    
    def __init__(self, compressed_dim: int, latent_dim: int, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        self.compressed_dim = compressed_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection for compressed patterns
        self.input_projection = nn.Linear(compressed_dim, hidden_dim)
        
        # Main LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Stabilization: LayerNorm on hidden before head (optional)
        self.hidden_norm = nn.LayerNorm(hidden_dim)

        # Output projection to latent vector (flattened)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)
        
        # Optimizer and scheduler for real-time training
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training state
        self.training_step = 0
        self.loss_history = deque(maxlen=200)
        
        # State tracking
        self.hidden_state = None
        self.cell_state = None
        
    def forward(self, compressed_patterns: torch.Tensor) -> torch.Tensor:
        """Predict next frame latent vector from compressed patterns"""
        # Project compressed patterns
        projected = self.input_projection(compressed_patterns)
        projected = projected.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # LSTM forward pass
        if self.hidden_state is None:
            lstm_out, (self.hidden_state, self.cell_state) = self.lstm(projected)
        else:
            # Detach hidden states to prevent backprop through previous steps' graphs
            self.hidden_state = self.hidden_state.detach()
            self.cell_state = self.cell_state.detach()
            lstm_out, (self.hidden_state, self.cell_state) = self.lstm(projected, (self.hidden_state, self.cell_state))
        
        # Get last output and normalize
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]
        last_output = self.hidden_norm(last_output)
        
        # Project to latent vector
        predicted_latent_vec = self.output_projection(last_output)  # [batch, latent_dim]
        
        return predicted_latent_vec
    
    def train_on_pair(self, compressed_prev: torch.Tensor, target_latent_vec: torch.Tensor, *,
                      teacher_forcing_ratio: float = 1.0,
                      prev_pred_latent: Optional[torch.Tensor] = None) -> float:
        """Train to predict target next-latent from previous compressed representation
        with scheduled sampling control.
        """
        self.train()
        
        # Choose input under scheduled sampling
        use_teacher = bool(torch.rand(()) < teacher_forcing_ratio)
        model_input = compressed_prev
        # If we later evolve to take prev_latent into account, this hook is ready
        _ = prev_pred_latent  # placeholder for future conditioning

        # Forward pass
        predicted_vec = self.forward(model_input)
        
        # Compute MSE loss
        loss = nn.MSELoss()(predicted_vec, target_latent_vec)
        # Tiny L2 on predicted delta magnitude to discourage overshoot if using deltas upstream
        loss = loss + 1e-6 * (predicted_vec.pow(2).mean())
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step(loss.item())
        
        # Track loss
        self.loss_history.append(loss.item())
        # Stash grad norm for external logging
        self.last_grad_norm = float(total_norm.detach().item()) if hasattr(total_norm, 'item') else float(total_norm)
        self.eval()
        return float(loss.item())
    
    def reset_state(self):
        """Reset LSTM hidden states"""
        self.hidden_state = None
        self.cell_state = None
