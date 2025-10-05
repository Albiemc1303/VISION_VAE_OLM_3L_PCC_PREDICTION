import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import cv2
import numpy as np
from PIL import Image
import time
from collections import deque
import os
import requests
from tqdm import tqdm

import importlib

AutoencoderKL = None


class VAEProcessor:
    """
    VAE processor for encoding/decoding frames to/from latents at 24fps.
    Uses AutoencoderKL and outputs a flattened latent vector for downstream LSTMs.
    """
    
    def __init__(self, vae_path: Path, config_path: Optional[Path] = None):
        self.vae_path = vae_path
        self.config_path = config_path or vae_path / "config.json"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Select precision: FP16 for CUDA, FP32 otherwise
        if self.device.type == 'cuda' and torch.cuda.is_available():
             # Use FP16 for faster inference on modern NVIDIA GPUs
             self.dtype = torch.float16
        else:
             self.dtype = torch.float32

        # Load configuration
        self.config = self._load_config()
        
        # Model parameters
        self.sample_size = self.config.get("sample_size", 512)
        self.latent_channels = self.config.get("latent_channels", 4)
        self.scaling_factor = self.config.get("scaling_factor", 0.18215)
        
        # Calculated latent dimensions
        self.latent_spatial_size = self.sample_size // 8
        self.latent_vec_size = self.latent_channels * self.latent_spatial_size * self.latent_spatial_size
        
        # Frame processing
        self.target_fps = 24
        self.frame_buffer = deque(maxlen=30)  # ~1.25 seconds buffer
        self.last_frame_time = 0
        self.frame_interval = 1.0 / self.target_fps
        
        # VAE model
        self.vae = None
        self._load_vae()
        
        # Last encoder posterior stats (pre-scaling)
        self.last_latent_mean_raw: Optional[torch.Tensor] = None
        self.last_latent_std_raw: Optional[torch.Tensor] = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load VAE configuration from JSON file"""
        if not self.config_path.exists():
            # For SD VAE, if config is missing, assume standard 512x512 config
            print(f"VAE config not found at {self.config_path}. Assuming standard SD-VAE config.")
            return {
                "sample_size": 512,
                "latent_channels": 4,
                "scaling_factor": 0.18215
            }
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _download_vae_model(self):
        """Download VAE model from Hugging Face if missing"""
        model_file = self.vae_path / "diffusion_pytorch_model.safetensors"

        if model_file.exists():
            return  # Already exists

        print("VAE model not found. Downloading from Hugging Face (sd-vae-ft-mse)...")

        self.vae_path.mkdir(parents=True, exist_ok=True)
        # Using the standard SD VAE
        url = "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors"

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(model_file, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading VAE") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"VAE model downloaded successfully to {model_file}")

        except Exception as e:
            if model_file.exists():
                model_file.unlink()
            raise RuntimeError(f"Failed to download VAE model: {e}")

    def _load_vae(self):
        """Load the VAE model"""
        # Attempt lazy import to avoid editor/static analysis errors when optional dependency is missing
        global AutoencoderKL
        if AutoencoderKL is None:
            try:
                diffusers = importlib.import_module("diffusers")
                AutoencoderKL = getattr(diffusers, "AutoencoderKL")
            except Exception:
                raise ImportError("diffusers library not installed. Install with: pip install diffusers")

        self._download_vae_model()

        try:
            # Load VAE from local files with determined precision
            self.vae = AutoencoderKL.from_pretrained(
                self.vae_path,
                torch_dtype=self.dtype
            )
            self.vae = self.vae.to(self.device)
            self.vae.eval()
            print(f"VAE loaded successfully on {self.device} with dtype {self.dtype}")
        except Exception as e:
            raise RuntimeError(f"Failed to load VAE: {e}")
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess camera frame for VAE encoding"""
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        frame = cv2.resize(frame, (self.sample_size, self.sample_size))
        
        # Normalize to [-1, 1] range and convert to tensor
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - 0.5) * 2.0
        
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device, dtype=self.dtype) # Convert to model dtype here
    
    def postprocess_frame(self, tensor: torch.Tensor) -> np.ndarray:
        """Postprocess VAE decoded tensor back to displayable frame"""
        tensor = tensor.float().squeeze(0).cpu() # Always convert back to float32 for processing
        
        # Denormalize
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy array
        frame = tensor.permute(1, 2, 0).numpy() * 255.0
        frame = frame.astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    
    def encode_frame(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """Encode frame to a single, flattened latent vector."""
        try:
            with torch.no_grad():
                # Preprocess frame (tensor is already converted to self.dtype)
                input_tensor = self.preprocess_frame(frame)
                
                # Encode to latents (use posterior mean for stability)
                posterior = self.vae.encode(input_tensor).latent_dist
                
                # Store raw posterior stats before scaling
                self.last_latent_mean_raw = posterior.mean.detach().float()
                self.last_latent_std_raw = posterior.std.detach().float()

                latents = posterior.mean
                latents = latents * self.scaling_factor
                
                # CRITICAL: Flatten the spatial latent tensor into a single vector
                # Latents are [1, C, H, W]. Flatten(start_dim=1) gives [1, C*H*W]
                flat_latent_vector = latents.flatten(start_dim=1) 
                
                return flat_latent_vector
        except Exception as e:
            print(f"Error encoding frame: {e}")
            return None
    
    def decode_latents(self, flat_latents: torch.Tensor) -> Optional[np.ndarray]:
        """Decode a single, flattened latent vector back to frame."""
        try:
            with torch.no_grad():
                # Reshape the flattened latent vector back to its 4D shape: [1, C, H, W]
                # flat_latents: [1, C*H*W]
                latents = flat_latents.view(
                    1, 
                    self.latent_channels, 
                    self.latent_spatial_size, 
                    self.latent_spatial_size
                )
                
                # Scale latents back
                latents = latents / self.scaling_factor
                
                # Ensure latents match model dtype
                latents = latents.to(dtype=self.dtype)
                
                # Decode to image
                decoded = self.vae.decode(latents).sample
                
                # Postprocess to frame
                frame = self.postprocess_frame(decoded)
                return frame
        except Exception as e:
            print(f"Error decoding latents: {e}")
            return None
    
    def should_process_frame(self) -> bool:
        """Check if enough time has passed to process next frame (24fps)"""
        current_time = time.time()
        if current_time - self.last_frame_time >= self.frame_interval:
            self.last_frame_time = current_time
            return True
        return False
    
    def process_camera_frame(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """Process camera frame and return latents if timing is correct"""
        if not self.should_process_frame():
            return None
        
        # Add to buffer (optional, depending on what the buffer is used for later)
        self.frame_buffer.append(frame)
        
        # Encode current frame
        latents = self.encode_frame(frame)
        return latents
    
    def get_latent_shape(self) -> Tuple[int, ...]:
        """Get the expected shape of the flattened latent tensor: [1, D]"""
        return (1, self.latent_vec_size)
    
    def get_latent_vector_size(self) -> int:
        """Get the total size of the flattened latent vector (D)"""
        return self.latent_vec_size

    def get_initial_latent_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Provides the expected initial mean and standard deviation for the 
        LSHCache normalization (before online updates begin).
        Returns: (initial_mu, initial_sigma), both [D]
        """
        # The latents are scaled by self.scaling_factor, but the VAE's intrinsic 
        # latent space has mean 0 and std 1. The LSHCache's purpose is to normalize 
        # the *final* vector.
        # Initial mu should be 0, initial sigma should be 1.
        # We need this to be the flattened size: [D]
        initial_mu = torch.zeros(self.latent_vec_size, device='cpu')
        initial_sigma = torch.ones(self.latent_vec_size, device='cpu')
        return initial_mu, initial_sigma
    
    def cleanup(self):
        """Clean up resources"""
        if self.vae is not None:
            del self.vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_last_encoder_stats(self) -> Dict[str, float]:
        """Return norms of the last encoder posterior mean/std (pre-scaling)."""
        stats: Dict[str, float] = {"mean_norm": 0.0, "std_norm": 0.0}
        try:
            if self.last_latent_mean_raw is not None:
                stats["mean_norm"] = float(self.last_latent_mean_raw.norm().item())
            if self.last_latent_std_raw is not None:
                stats["std_norm"] = float(self.last_latent_std_raw.norm().item())
        except Exception:
            pass
        return stats