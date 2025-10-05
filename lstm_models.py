import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any, List
from collections import deque
import math


class PatternLSTM(nn.Module):
    """Frozen LSTM for pattern detection in latent sequences"""
    
    def __init__(self, latent_vec_size: int, hidden_dim: int = 512, num_layers: int = 2, buffer_size: int = 16):
        super().__init__()
        self.latent_vec_size = latent_vec_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.buffer_size = buffer_size
        
        # Input dimension is the size of the *flattened* latent vector
        self.input_dim = latent_vec_size
        
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
        # Stores [1, latent_vec_size] tensors
        self.latent_buffer = deque(maxlen=buffer_size)
        
    def freeze_weights(self):
        """Freeze all model parameters"""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
    
    def add_latent(self, latent: torch.Tensor):
        """Add latent to rolling buffer"""
        # Ensure latent is [1, D] where D is the expected vector size
        if latent.numel() != self.latent_vec_size:
            # Handle possible spatial input by flattening
            flattened = latent.view(1, -1)
            if flattened.size(1) != self.latent_vec_size:
                raise ValueError(f"Latent vector size mismatch: Expected {self.latent_vec_size}, got {flattened.size(1)} after flattening.")
            self.latent_buffer.append(flattened)
        else:
            self.latent_buffer.append(latent.view(1, -1)) # Ensure [1, D] format
    
    def get_latent_sequence(self) -> torch.Tensor:
        """Returns the full sequence of latents [1, seq_len, D]"""
        if len(self.latent_buffer) == 0:
            return torch.zeros(1, 0, self.latent_vec_size, device=next(self.parameters()).device)
            
        # Convert buffer to sequence tensor (current: [seq_len, 1, input_dim])
        sequence = torch.stack(list(self.latent_buffer))
        # Reorder to batch_first: [1, seq_len, input_dim]
        sequence = sequence.transpose(0, 1)
        return sequence
        
    def detect_patterns(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Detect patterns in the latent sequence and return last output and full sequence outputs.

        Returns:
            last_output: [batch=1, seq=1, hidden_dim]
            full_output: [batch=1, seq_len, hidden_dim]
        """
        sequence = self.get_latent_sequence()
        seq_len = sequence.size(1)
        
        if seq_len < 2:
            # Return zero tensors if insufficient data
            zero_last = torch.zeros(1, 1, self.hidden_dim, device=sequence.device)
            zero_full = torch.zeros(1, seq_len if seq_len > 0 else 1, self.hidden_dim, device=sequence.device)
            return zero_last, zero_full
        
        with torch.no_grad():
            # LSTM forward (batch_first=True)
            # lstm_output: [1, seq_len, hidden_dim]
            lstm_output, _ = self.lstm(sequence) 
            
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
        Accepts patterns in [batch, seq_len, pattern_dim] (preferred) or [seq_len, batch, pattern_dim]."""
        
        # Standardize input to [batch, seq_len, pattern_dim]
        if patterns.dim() == 2:
            patterns = patterns.unsqueeze(0) # [seq_len, H] -> [1, seq_len, H]
        if patterns.size(0) != 1 and patterns.size(1) == 1:
            patterns = patterns.transpose(0, 1) # [seq_len, batch, H] -> [batch, seq_len, H]

        batch_size = patterns.size(0)
        seq_len = patterns.size(1)
        pattern_dim = patterns.size(2)

        # Padding/Trimming for consistency
        if pattern_dim != self.pattern_dim:
            if pattern_dim > self.pattern_dim:
                patterns = patterns[:, :, :self.pattern_dim]
            else:
                padding = torch.zeros(batch_size, seq_len, self.pattern_dim - pattern_dim,
                                      device=patterns.device, dtype=patterns.dtype)
                patterns = torch.cat([patterns, padding], dim=2)
        
        # LSTM forward pass
        # lstm_out: [batch, seq_len, compressed_dim]
        # hidden, cell: [num_layers, batch, compressed_dim]
        lstm_out, (hidden, cell) = self.lstm(patterns)
        
        # Use last hidden state of the *top layer*
        last_hidden = hidden[-1]  # [batch, compressed_dim]
        
        # Project and activate
        compressed = self.output_projection(last_hidden)
        compressed = self.activation(compressed)
        
        return compressed
    
    def compress_patterns(self, patterns: torch.Tensor) -> torch.Tensor:
        """Compress patterns and optionally train."""
        do_train = (self.training_step % 5 == 0)
        if do_train:
            self.train()
            compressed = self.forward(patterns)
            self._train_step(patterns, compressed)
            self.eval()
            
        # Always run the final prediction in eval mode (no_grad)
        with torch.no_grad():
            compressed_out = self.forward(patterns)
            
        self.training_step += 1
        return compressed_out
    
    def _train_step(self, patterns: torch.Tensor, compressed: torch.Tensor):
        """Perform one training step with anti-collapse objective."""
        self.train()
        
        # Standardize patterns to [batch, seq_len, H]
        if patterns.dim() != 3:
            patterns = patterns.unsqueeze(0)
        if patterns.size(0) != 1 and patterns.size(1) == 1:
            patterns = patterns.transpose(0, 1)

        seq_len = patterns.size(1)

        # Target: The last pattern vector in the sequence
        target_pattern = patterns[:, -1, :] 

        # Map compressed to pattern space to predict the target pattern
        predicted_target = self.reconstruction_head(compressed)  # [batch, H]

        # Loss 1: Prediction MSE
        loss_pred = nn.MSELoss()(predicted_target, target_pattern)
        
        # Loss 2: Variance encouragement (Anti-Collapse)
        var = torch.var(compressed, dim=1).mean() # Variance of the compressed vector across batch
        loss_var = -0.001 * var
        
        # Total Loss
        loss = loss_pred + loss_var
        
        # Regularization (L2)
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


class CentralLSTM(nn.Module):
    """
    Central LSTM for next frame prediction.
    Predicts the latent DELTA (change) rather than the absolute value for stability.
    Input is the concatenation of the current latent vector (z_t) and the compressed pattern vector (c_t).
    """
    
    def __init__(self, compressed_dim: int, latent_dim: int, hidden_dim: int = 512, num_layers: int = 3):
        super().__init__()
        self.compressed_dim = compressed_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # CRITICAL CHANGE: Input dimension is now the concatenation of latent_dim + compressed_dim
        self.combined_input_dim = latent_dim + compressed_dim
        
        # Input projection for combined vector [z_t, c_t]
        self.input_projection = nn.Linear(self.combined_input_dim, hidden_dim)
        
        # Main LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        # Stabilization: LayerNorm on hidden before head
        self.hidden_norm = nn.LayerNorm(hidden_dim)

        # Output projection to latent DELTA vector (flattened)
        self.output_projection = nn.Linear(hidden_dim, latent_dim)
        
        # Optimizer and scheduler for real-time training
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Training state
        self.training_step = 0
        self.loss_history = deque(maxlen=200)
        self.last_grad_norm = 0.0
        
        # State tracking
        self.hidden_state: Optional[torch.Tensor] = None
        self.cell_state: Optional[torch.Tensor] = None
        
    def forward(self, combined_input: torch.Tensor) -> torch.Tensor:
        """
        Predict next latent DELTA vector.
        
        Args:
            combined_input: The concatenated vector [z_t, c_t] of size [batch, latent_dim + compressed_dim]
            
        Returns:
            predicted_delta_vec: The predicted delta vector [batch, latent_dim]
        """
        # Ensure combined_input is [batch, combined_input_dim]
        if combined_input.ndim == 1:
            combined_input = combined_input.unsqueeze(0)
            
        # Error check (collaborative fact check)
        if combined_input.size(-1) != self.combined_input_dim:
            # Gently point out the error
            raise ValueError(f"Input dimension mismatch in CentralLSTM forward: Expected {self.combined_input_dim} (latent+compressed), but got {combined_input.size(-1)}. This needs to be checked in the AIPipeline to ensure correct concatenation.")
            
        # Project combined input
        projected = self.input_projection(combined_input)
        projected = projected.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # LSTM forward pass (stateful)
        if self.hidden_state is None:
            lstm_out, (self.hidden_state, self.cell_state) = self.lstm(projected)
        else:
            # Detach hidden states for BPTT control
            h_in = self.hidden_state.detach()
            c_in = self.cell_state.detach()
            lstm_out, (self.hidden_state, self.cell_state) = self.lstm(projected, (h_in, c_in))
            
        # Get last output and normalize
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]
        last_output = self.hidden_norm(last_output)
        
        # Project to latent DELTA vector
        predicted_delta_vec = self.output_projection(last_output)  # [batch, latent_dim]
        
        return predicted_delta_vec
    
    def predict_next_latent(self, combined_input: torch.Tensor, current_latent: torch.Tensor) -> torch.Tensor:
        """Inference step: Predict delta and add to current latent (z_t+1 = z_t + delta_z)."""
        with torch.no_grad():
            self.eval()
            # Predict delta
            predicted_delta = self.forward(combined_input)
            # Compute next latent: z_t+1 = z_t + delta
            # The current_latent (z_t) is the first part of the combined_input vector, 
            # but we pass it explicitly here for clarity and safety.
            predicted_next_latent = current_latent + predicted_delta
            return predicted_next_latent
    
    def train_on_pair(self, combined_input_prev: torch.Tensor, target_delta: torch.Tensor, *,
                      l2_lambda: float = 1e-6) -> float:
        """
        Train to predict target latent DELTA (target_delta = z_t - z_{t-1}) 
        from the previous combined state (combined_input_prev = [z_{t-1}, c_{t-1}]).
        
        Args:
            combined_input_prev: The concatenated vector [z_{t-1}, c_{t-1}]
            target_delta: The true difference vector z_t - z_{t-1}
            l2_lambda: L2 regularization weight on the predicted delta magnitude.
            
        Returns:
            The resulting MSE loss value.
        """
        self.train()
        
        # 1. Forward Pass
        # Predict delta_z_t = z_t - z_{t-1}
        predicted_delta = self.forward(combined_input_prev)
        
        # 2. Compute MSE loss on DELTA
        loss = nn.MSELoss()(predicted_delta, target_delta)
        
        # L2 on predicted delta magnitude
        loss = loss + l2_lambda * (predicted_delta.pow(2).mean())
        
        # 3. Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step(loss.item())
        
        # 4. Track Loss
        self.loss_history.append(loss.item())
        self.last_grad_norm = float(total_norm.detach().item()) if hasattr(total_norm, 'item') else float(total_norm)
        self.eval()
        return float(loss.item())
        
    def reset_state(self):
        """Reset LSTM hidden states"""
        self.hidden_state = None
        self.cell_state = None