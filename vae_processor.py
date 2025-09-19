import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import cv2
import numpy as np
from PIL import Image
import time
from collections import deque
import os
import requests
from tqdm import tqdm

try:
    from diffusers import AutoencoderKL
except ImportError:
    AutoencoderKL = None


class VAEProcessor:
    """VAE processor for encoding/decoding frames to/from latents at 24fps"""
    
    def __init__(self, vae_path: Path, config_path: Optional[Path] = None):
        self.vae_path = vae_path
        self.config_path = config_path or vae_path / "config.json"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        self.config = self._load_config()
        
        # Model parameters
        self.sample_size = self.config.get("sample_size", 512)
        self.latent_channels = self.config.get("latent_channels", 4)
        self.scaling_factor = self.config.get("scaling_factor", 0.18215)
        
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
            raise FileNotFoundError(f"VAE config not found at {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def _download_vae_model(self):
        """Download VAE model from Hugging Face if missing"""
        model_file = self.vae_path / "diffusion_pytorch_model.safetensors"

        if model_file.exists():
            return  # Already exists

        print("VAE model not found. Downloading from Hugging Face...")

        # Ensure directory exists
        self.vae_path.mkdir(parents=True, exist_ok=True)

        # Download URL for Stable Diffusion VAE
        url = "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors"

        try:
            # Download with progress bar
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
            # Clean up partial download
            if model_file.exists():
                model_file.unlink()
            raise RuntimeError(f"Failed to download VAE model: {e}")

    def _load_vae(self):
        """Load the VAE model"""
        if AutoencoderKL is None:
            raise ImportError("diffusers library not installed. Install with: pip install diffusers")

        # Check and download VAE model if needed
        self._download_vae_model()

        try:
            # Load VAE from local files - use float32 for compatibility
            self.vae = AutoencoderKL.from_pretrained(
                self.vae_path,
                torch_dtype=torch.float32
            )
            self.vae = self.vae.to(self.device)
            self.vae.eval()
            print(f"VAE loaded successfully on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load VAE: {e}")
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess camera frame for VAE encoding"""
        # Convert BGR to RGB
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to VAE input size
        frame = cv2.resize(frame, (self.sample_size, self.sample_size))
        
        # Normalize to [-1, 1] range
        frame = frame.astype(np.float32) / 255.0
        frame = (frame - 0.5) * 2.0
        
        # Convert to tensor and add batch dimension
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def postprocess_frame(self, tensor: torch.Tensor) -> np.ndarray:
        """Postprocess VAE decoded tensor back to displayable frame"""
        # Remove batch dimension and move to CPU
        tensor = tensor.squeeze(0).cpu()
        
        # Denormalize from [-1, 1] to [0, 255]
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)
        tensor = tensor * 255.0
        
        # Convert to numpy array
        frame = tensor.permute(1, 2, 0).numpy().astype(np.uint8)
        
        # Convert RGB to BGR for OpenCV display
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame
    
    def encode_frame(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """Encode frame to latents using VAE encoder"""
        try:
            with torch.no_grad():
                # Preprocess frame
                input_tensor = self.preprocess_frame(frame)
                
                # Ensure input tensor matches model dtype
                if hasattr(self.vae, 'dtype'):
                    input_tensor = input_tensor.to(dtype=self.vae.dtype)
                else:
                    input_tensor = input_tensor.to(dtype=torch.float32)
                
                # Encode to latents (use posterior mean for stability)
                posterior = self.vae.encode(input_tensor).latent_dist
                # Store raw posterior stats before scaling
                try:
                    self.last_latent_mean_raw = posterior.mean.detach()
                    self.last_latent_std_raw = posterior.std.detach()
                except Exception:
                    self.last_latent_mean_raw = None
                    self.last_latent_std_raw = None
                latents = posterior.mean
                latents = latents * self.scaling_factor
                
                return latents
        except Exception as e:
            print(f"Error encoding frame: {e}")
            return None
    
    def decode_latents(self, latents: torch.Tensor) -> Optional[np.ndarray]:
        """Decode latents back to frame using VAE decoder"""
        try:
            with torch.no_grad():
                # Scale latents back
                latents = latents / self.scaling_factor
                
                # Ensure latents match model dtype
                if hasattr(self.vae, 'dtype'):
                    latents = latents.to(dtype=self.vae.dtype)
                else:
                    latents = latents.to(dtype=torch.float32)
                
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
        
        # Add to buffer
        self.frame_buffer.append(frame)
        
        # Encode current frame
        latents = self.encode_frame(frame)
        return latents
    
    def get_latent_shape(self) -> Tuple[int, ...]:
        """Get the expected shape of latent tensors"""
        # For Stable Diffusion VAE: [batch, latent_channels, height/8, width/8]
        latent_size = self.sample_size // 8
        return (1, self.latent_channels, latent_size, latent_size)
    
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
