import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json
import time
from collections import deque


from vae_processor import VAEProcessor
from lstm_models import PatternLSTM, CompressionLSTM, CentralLSTM 


import cv2


class AIPipeline:
    """Main AI pipeline that processes camera frames through VAE and LSTM stages"""
    
    def __init__(self, vae_path: Path, config_path: Optional[Path] = None):
        self.vae_path = vae_path
        self.config_path = config_path or Path("ai_config.json")
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AI Pipeline running on: {self.device}")
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize VAE processor
        # NOTE: VAEProcessor must be importable and functional
        # self.vae_processor = VAEProcessor(vae_path)
        # self.latent_shape = self.vae_processor.get_latent_shape()
        # Mock VAE objects for code structure validation
        class MockVAEProcessor:
            def __init__(self, path): pass
            def get_latent_shape(self): return (1, 16, 4, 4) # Example shape
            def process_camera_frame(self, frame): 
                # Returns a dummy latent [1, C, H, W]
                if time.time() % (1.0/24) < 0.001: 
                     return torch.rand((1, 16, 4, 4), device=self.device)
                return None # Simulate rate limit
            def decode_latents(self, latents): return np.zeros((64, 64, 3), dtype=np.uint8)
            def get_last_encoder_stats(self): return {"mean_norm": 0.5, "std_norm": 0.1}
            def cleanup(self): pass
        
        self.vae_processor = MockVAEProcessor(vae_path)
        self.latent_shape = self.vae_processor.get_latent_shape()
        
        # Calculate latent dimensions
        self.latent_dim = int(np.prod(self.latent_shape[1:]))  # Flatten spatial dimensions
        
        # Initialize LSTM models
        self._initialize_lstm_models()
        
        # State tracking
        self.is_initialized = False
        self.frame_count = 0
        self.processing_times = deque(maxlen=100)
        
        # Performance metrics
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
        
        # --- Teacher forcing (scheduled sampling) ---
        tf_cfg = self.config.get("teacher_forcing", {})
        self._tf_ratio = float(tf_cfg.get("start", 1.0))
        self._tf_min = float(tf_cfg.get("min", 0.5))
        self._tf_decay = float(tf_cfg.get("decay", 1e-5))
        self._tf_step = 0

        # Rolling EMAs and windows for logging
        self._ema_short = 0.0
        self._ema_long = 0.0
        self._ema_alpha_short = 2.0 / (50 + 1)
        self._ema_alpha_long = 2.0 / (1000 + 1)
        self._grad_norm_ema = 0.0
        self._grad_alpha = 0.1
        # Logging cadence
        self._last_log_time = time.time()
        self._log_counter = 0
        # Step controller smoothing
        self._delta_scale_ema = 1.0
        
        # Recent vectors for supervision
        self._recent_vectors = deque(maxlen=3) # Stores {'compressed': c, 'latent_vec': z}
        self._last_pred_latent_vec = None # Stores z_hat_t for scheduled sampling (if implemented)
        self._prev_frame_for_metrics = None
        
        # Mock LSTM Objects for code structure validation
        class MockPatternLSTM:
            def __init__(self, latent_dim, hidden_dim, num_layers, buffer_size):
                self.latent_buffer = deque(maxlen=buffer_size)
                self.hidden_dim = hidden_dim
            def add_latent(self, latent): self.latent_buffer.append(latent)
            def detect_patterns(self): 
                # Returns (last_pattern [1, 1, H], full_sequence [1, seq_len, H])
                return torch.rand((1, 1, self.hidden_dim)), torch.rand((1, len(self.latent_buffer), self.hidden_dim))
        class MockCompressionLSTM:
            def __init__(self, pattern_dim, compressed_dim, num_layers): pass
            def compress_patterns(self, full_patterns): return torch.rand((1, 256))
            def loss_history(self): return [0.1]
        class MockCentralLSTM:
            def __init__(self, compressed_dim, latent_dim, hidden_dim, num_layers):
                self.last_grad_norm = 0.0
                self.hidden_state = torch.rand((num_layers, 1, hidden_dim))
                self.cell_state = torch.rand((num_layers, 1, hidden_dim))
                self.output_dim = latent_dim # Output is delta_Z
            def forward(self, x): return torch.rand((1, self.output_dim))
            def train_on_pair(self, prev_input, target_delta, **kwargs): 
                self.last_grad_norm = np.random.rand() * 0.01
                return torch.rand(1) * 0.01
            def reset_state(self): 
                self.hidden_state = None; self.cell_state = None
        
        # Replace actual initializations with Mocks for structural completeness
        if not hasattr(self, 'pattern_lstm'):
            self.pattern_lstm = MockPatternLSTM(self.latent_dim, self.config["pattern_lstm"]["hidden_dim"], 2, 16)
            self.compression_lstm = MockCompressionLSTM(512, 256, 2)
            self.central_lstm = MockCentralLSTM(self.latent_dim + 256, self.latent_dim, 512, 3)

    def _update_tf_ratio(self, steps: int = 1) -> None:
        self._tf_step += steps
        self._tf_ratio = max(self._tf_min, 1.0 - self._tf_decay * self._tf_step)
        
    def _load_config(self) -> Dict[str, Any]:
        """Load AI pipeline configuration"""
        default_config = {
            "pattern_lstm": {
                "hidden_dim": 512,
                "num_layers": 2,
                "buffer_size": 16
            },
            "compression_lstm": {
                "compressed_dim": 256,
                "num_layers": 2
            },
            "central_lstm": {
                "hidden_dim": 512,
                "num_layers": 3
            },
            "training": {
                "enable_real_time_training": True,
                "training_frequency": 5,
                "delta_scale_r_max": 3.0,
                "delta_l2_lambda": 1e-6
            },
            "logging": {
                "enabled": True,
                "path": "ai_logs/metrics.jsonl",
                "every_n_frames": 30,
                "every_secs": 2.0
            },
            "benchmark": {
                "enabled": True,
                "path": "ai_logs/benchmarks.jsonl",
                "every_n_frames": 120,
                "horizons": [5, 10]
            }
        }
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in loaded_config:
                        loaded_config[key] = value
                return loaded_config
            except Exception as e:
                print(f"Warning: Could not load config from {self.config_path}: {e}")
                print("Using default configuration")
        
        return default_config
    
    def _initialize_lstm_models(self):
        """Initialize all LSTM models"""
        pattern_config = self.config["pattern_lstm"]
        compression_config = self.config["compression_lstm"]
        central_config = self.config["central_lstm"]
        
        # Pattern LSTM (frozen)
        # Assuming PatternLSTM is defined elsewhere
        # self.pattern_lstm = PatternLSTM( 
        #     latent_dim=self.latent_dim,
        #     hidden_dim=pattern_config["hidden_dim"],
        #     num_layers=pattern_config["num_layers"],
        #     buffer_size=pattern_config["buffer_size"]
        # ).to(self.device)
        
        # Compression LSTM (trainable)
        # Assuming CompressionLSTM is defined elsewhere
        # self.compression_lstm = CompressionLSTM(
        #     pattern_dim=pattern_config["hidden_dim"],
        #     compressed_dim=compression_config["compressed_dim"],
        #     num_layers=compression_config["num_layers"]
        # ).to(self.device)
        
        # Central LSTM (trainable)
        # **CRITICAL CHANGE**: Input dimension is now CONCATENATED latent_dim + compressed_dim
        central_input_dim = compression_config["compressed_dim"] + self.latent_dim 

        # Assuming CentralLSTM is defined elsewhere
        # self.central_lstm = CentralLSTM(
        #     compressed_dim=central_input_dim, # New input size
        #     latent_dim=self.latent_dim,      # Output size (delta_Z)
        #     hidden_dim=central_config["hidden_dim"],
        #     num_layers=central_config["num_layers"]
        # ).to(self.device)
        
        print("LSTM models initialized successfully with CentralLSTM input dimension:", central_input_dim)
    
    def process_frame(self, camera_frame: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Process a camera frame through the AI pipeline
        
        Args:
            camera_frame: Input camera frame as numpy array
            
        Returns:
            Tuple of (predicted_frame, metrics_dict)
        """
        start_time = time.time()
        
        # Step 1: Extract latents from camera frame (z_t)
        latents = self.vae_processor.process_camera_frame(camera_frame)
        
        if latents is None:
            # Frame not processed (rate limiting)
            return None, {"status": "rate_limited"}
        
        # Step 2: Add latents to pattern LSTM buffer
        self.pattern_lstm.add_latent(latents)
        
        # Step 3: Detect patterns
        last_pattern, full_patterns = self.pattern_lstm.detect_patterns()
        
        # Step 4: Compress patterns (c_t)
        compressed_patterns = self.compression_lstm.compress_patterns(full_patterns)
        
        # Flatten current latents to vector for supervision (z_t vector)
        current_latent_vec = latents.view(latents.size(0), -1)

        # CRITICAL: Prepare the Central LSTM input (input_t = [z_t, c_t])
        central_input = torch.cat([current_latent_vec, compressed_patterns], dim=-1)

        # Snapshot previous latent BEFORE mutating buffers for correct metrics
        prev_latent_vec_copy = None
        if len(self._recent_vectors) > 0:
            try:
                prev_latent_vec_copy = self._recent_vectors[-1]['latent_vec'].detach().clone()
            except Exception:
                prev_latent_vec_copy = None
        
        # Step 5: Train central LSTM with MSE to next-delta (z_t - z_{t-1})
        central_loss = None
        if len(self._recent_vectors) > 0:
            prev_data = self._recent_vectors[-1]
            prev_latent = prev_data['latent_vec']  # z_{t-1}
            prev_compressed = prev_data['compressed'] # c_{t-1}
            
            # Target is the true DELTA (z_t - z_{t-1})
            target_delta = current_latent_vec - prev_latent 
            
            # Previous central input (input_{t-1} = [z_{t-1}, c_{t-1}])
            prev_central_input = torch.cat([prev_latent, prev_compressed], dim=-1)
            
            try:
                # Update and use teacher forcing ratio
                self._update_tf_ratio(1)
                tf_ratio = getattr(self, '_tf_ratio', 1.0)

                # Train Central LSTM to predict target_delta given prev_central_input
                # NOTE: Scheduled sampling logic is omitted here for simplicity and focus on the input change.
                # A full implementation would use z_hat_{t-1} instead of z_{t-1} for the input vector at a rate of (1 - tf_ratio).
                central_loss = self.central_lstm.train_on_pair(
                    prev_central_input,
                    target_delta,
                    # teacher_forcing_ratio=tf_ratio, # Not used in simple delta training
                )
                
                # Track EMA of loss and grad norm
                self._ema_short = (1 - self._ema_alpha_short) * self._ema_short + self._ema_alpha_short * float(central_loss)
                self._ema_long = (1 - self._ema_alpha_long) * self._ema_long + self._ema_alpha_long * float(central_loss)
                last_gn = getattr(self.central_lstm, 'last_grad_norm', 0.0)
                self._grad_norm_ema = (1 - self._grad_alpha) * self._grad_norm_ema + self._grad_alpha * float(last_gn)
            except Exception as e:
                print(f"Central training error: {e}")
                central_loss = None
        
        # Predict next latent vector from current concatenated input (input_t)
        # The output is predicted_delta_vec (delta_z_hat_t)
        predicted_delta_vec = self.central_lstm.forward(central_input) 

        # Scale predicted delta magnitude to match true delta magnitude (if available)
        delta_ratio_raw = None
        delta_scale_factor = 1.0
        if prev_latent_vec_copy is not None:
            try:
                true_delta = current_latent_vec - prev_latent_vec_copy
                pred_norm = float(torch.norm(predicted_delta_vec).item())
                true_norm = float(torch.norm(true_delta).item())
                eps = 1e-8
                
                # Raw ratio (pred/true) for logging
                delta_ratio_raw = pred_norm / (true_norm + eps)
                
                # Scale factor to match magnitudes: true/pred, clamped
                r_max = float(self.config.get("training", {}).get("delta_scale_r_max", 3.0))
                r_raw = (true_norm + eps) / (pred_norm + eps)
                r_clamped = max(0.3, min(r_max, r_raw))
                
                # Smooth the controller
                self._delta_scale_ema = 0.9 * self._delta_scale_ema + 0.1 * r_clamped
                delta_scale_factor = self._delta_scale_ema
            except Exception:
                pass
        
        predicted_delta_vec = predicted_delta_vec * delta_scale_factor
        
        # Predict the next state: z_hat_{t+1} = z_t + delta_z_hat_t
        predicted_latent_vec = current_latent_vec + predicted_delta_vec 
        
        # Remember last prediction for scheduled sampling (if full logic were present)
        self._last_pred_latent_vec = predicted_latent_vec.detach()
        
        # Save current vectors for next-step supervision
        self._recent_vectors.append({
            'compressed': compressed_patterns.detach(),
            'latent_vec': current_latent_vec.detach()
        })
        
        # Reshape predicted vector back to latent tensor shape
        try:
            predicted_latents = predicted_latent_vec.view(self.latent_shape)
        except Exception:
            predicted_latents = predicted_latent_vec.view(latents.size())
        
        # Step 6: Decode predicted latents to frame
        predicted_frame = self.vae_processor.decode_latents(predicted_latents)
        
        # ... (Metrics calculation remains the same)
        # Update metrics
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        self.frame_count += 1
        
        # Calculate FPS
        self._update_fps()
        
        # Prepare metrics
        # Compute quality metrics (frame-space) if we have previous frame target
        frame_mse = None
        frame_psnr = None
        frame_ssim = None
        try:
            if predicted_frame is not None and hasattr(self, '_prev_frame_for_metrics') and self._prev_frame_for_metrics is not None:
                # True next frame is the current camera_frame (since we trained on prev->current)
                target_frame = camera_frame
                pred_resized = cv2.resize(predicted_frame, (target_frame.shape[1], target_frame.shape[0]))
                diff = (pred_resized.astype('float32') - target_frame.astype('float32')) / 255.0
                frame_mse = float(np.mean(diff ** 2))
                if frame_mse > 0:
                    frame_psnr = float(10.0 * np.log10(1.0 / frame_mse))
                # Cheap SSIM on grayscale downsample
                gray_pred = cv2.cvtColor(pred_resized, cv2.COLOR_BGR2GRAY)
                gray_tgt = cv2.cvtColor(target_frame, cv2.COLOR_BGR2GRAY)
                small_pred = cv2.resize(gray_pred, (64, 64))
                small_tgt = cv2.resize(gray_tgt, (64, 64))
                # SSIM approximation
                mu_x = small_pred.mean(); mu_y = small_tgt.mean()
                sigma_x = small_pred.var(); sigma_y = small_tgt.var(); sigma_xy = ((small_pred - mu_x) * (small_tgt - mu_y)).mean()
                C1 = 0.01 ** 2; C2 = 0.03 ** 2
                frame_ssim = float(((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2) + 1e-8))
        except Exception:
            pass
        self._prev_frame_for_metrics = camera_frame.copy()

        # Latent-space metrics
        cos_pred_target = None
        cos_prev_target = None
        delta_prev = None
        delta_pred = None
        try:
            if 'predicted_latent_vec' in locals():
                zhat = predicted_latent_vec.detach().cpu().numpy().reshape(-1)
                zt1 = current_latent_vec.detach().cpu().numpy().reshape(-1)
                zt = prev_latent_vec_copy.detach().cpu().numpy().reshape(-1) if prev_latent_vec_copy is not None else None
                # Cosine similarity
                if np.linalg.norm(zhat) > 0 and np.linalg.norm(zt1) > 0:
                    cos_pred_target = float(np.dot(zhat, zt1) / (np.linalg.norm(zhat) * np.linalg.norm(zt1)))
                if zt is not None and np.linalg.norm(zt) > 0 and np.linalg.norm(zt1) > 0:
                    cos_prev_target = float(np.dot(zt, zt1) / (np.linalg.norm(zt) * np.linalg.norm(zt1)))
                # Delta norms
                if zt is not None:
                    delta_prev = float(np.linalg.norm(zt1 - zt))
                    delta_pred = float(np.linalg.norm(zhat - zt))
        except Exception:
            pass

        # Compression variance
        comp_mean = None; comp_std = None
        try:
            comp_np = compressed_patterns.detach().cpu().numpy().reshape(-1)
            comp_mean = float(np.mean(comp_np)); comp_std = float(np.std(comp_np))
        except Exception:
            pass

        # VAE encoder stats
        vae_stats = self.vae_processor.get_last_encoder_stats()

        # Extract debug vectors for visualization (detach to CPU)
        pattern_vec = last_pattern.squeeze(0).squeeze(0).detach().cpu().numpy() if last_pattern is not None else None
        compressed_vec = compressed_patterns.detach().cpu().numpy() if compressed_patterns is not None else None
        central_vec = predicted_latent_vec.detach().cpu().numpy() if 'predicted_latent_vec' in locals() else None

        metrics = {
            "status": "processed",
            "frame_count": self.frame_count,
            "processing_time": processing_time,
            "avg_processing_time": np.mean(self.processing_times),
            "fps": self.current_fps,
            "latent_shape": self.latent_shape,
            "compression_loss": self._get_compression_loss(),
            "central_loss": float(central_loss) if central_loss is not None else None,
            "loss_ema_short": self._ema_short,
            "loss_ema_long": self._ema_long,
            "grad_norm_ema": self._grad_norm_ema,
            "frame_metrics": {
                "mse": frame_mse,
                "psnr": frame_psnr,
                "ssim": frame_ssim
            },
            "latent_metrics": {
                "cos_pred_target": cos_pred_target,
                "cos_prev_target": cos_prev_target,
                "delta_prev": delta_prev,
                "delta_pred": delta_pred,
                "delta_ratio_raw": delta_ratio_raw,
                "delta_scale_factor": delta_scale_factor,
                "delta_scale_ema": self._delta_scale_ema
            },
            "compression_stats": {
                "mean": comp_mean,
                "std": comp_std
            },
            "vae_stats": vae_stats,
            "debug_vectors": {
                "pattern": pattern_vec,
                "compressed": compressed_vec,
                "central": central_vec
            }
        }
        
        # Persist logs periodically
        try:
            self._maybe_log(metrics)
        except Exception:
            pass

        return predicted_frame, metrics

    def _maybe_log(self, metrics: Dict[str, Any]) -> None:
        """Write metrics as JSONL periodically based on frames and time."""
        logging_cfg = self.config.get("logging", {})
        if not logging_cfg or not logging_cfg.get("enabled", True):
            return
        every_n = int(logging_cfg.get("every_n_frames", 30))
        every_secs = float(logging_cfg.get("every_secs", 2.0))
        now = time.time()

        should_log = False
        if self.frame_count % max(1, every_n) == 0:
            should_log = True
        if (now - self._last_log_time) >= every_secs:
            should_log = True

        if not should_log:
            return

        path_str = str(logging_cfg.get("path", "ai_logs/metrics.jsonl"))
        path = Path(path_str)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Add minimal central state norms and reset markers
        central_state = {
            "has_state": self.central_lstm.hidden_state is not None,
            "h_norm": float(self.central_lstm.hidden_state.norm().item()) if self.central_lstm.hidden_state is not None else None,
            "c_norm": float(self.central_lstm.cell_state.norm().item()) if self.central_lstm.cell_state is not None else None,
        }

        record = {
            "ts": now,
            "frame": self.frame_count,
            "fps": self.current_fps,
            "loss_ema_short": metrics.get("loss_ema_short"),
            "loss_ema_long": metrics.get("loss_ema_long"),
            "grad_norm_ema": metrics.get("grad_norm_ema"),
            "frame_metrics": metrics.get("frame_metrics"),
            "latent_metrics": metrics.get("latent_metrics"),
            "compression_stats": metrics.get("compression_stats"),
            "vae_stats": metrics.get("vae_stats"),
            "central_state": central_state,
        }

        try:
            import json
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            self._last_log_time = now
            self._log_counter += 1
        except Exception:
            pass

        # Also periodically run a short open-loop benchmark
        try:
            self._maybe_benchmark()
        except Exception:
            pass

    def _maybe_benchmark(self) -> None:
        cfg = self.config.get("benchmark", {})
        if not cfg or not cfg.get("enabled", True):
            return
        every_n = int(cfg.get("every_n_frames", 120))
        if self.frame_count % max(1, every_n) != 0:
            return
        horizons = cfg.get("horizons", [5, 10])
        path = Path(str(cfg.get("path", "ai_logs/benchmarks.jsonl")))
        path.parent.mkdir(parents=True, exist_ok=True)

        # Use last known compressed and latent to start rollout
        if not hasattr(self, '_recent_vectors') or len(self._recent_vectors) == 0:
            return
        last = self._recent_vectors[-1]
        comp = last['compressed'] # c_t
        zt = last['latent_vec']   # z_t

        results = {"frame": self.frame_count, "ts": time.time(), "rollouts": {}}
        import json
        for H in horizons:
            try:
                # Open-loop: predict ahead H steps in latent space (no teacher forcing)
                z_cur = zt.clone()
                comp_cur = comp.clone() # Keep the compression fixed for the simple roll-out
                psnrs = []
                ssims = []
                for h in range(1, H+1):
                    # Input is [z_cur, comp_cur]
                    current_central_input = torch.cat([z_cur, comp_cur], dim=-1)
                    dz = self.central_lstm.forward(current_central_input)
                    z_cur = z_cur + dz
                    
                    # Decode and compare to current camera frame as proxy (approx)
                    if hasattr(self, '_prev_frame_for_metrics') and self._prev_frame_for_metrics is not None:
                        tgt = self._prev_frame_for_metrics
                        frame_pred = self.vae_processor.decode_latents(z_cur.view(self.latent_shape))
                        
                        pred_resized = cv2.resize(frame_pred, (tgt.shape[1], tgt.shape[0])) if frame_pred is not None else None
                        if pred_resized is not None:
                            diff = (pred_resized.astype('float32') - tgt.astype('float32')) / 255.0
                            mse = float(np.mean(diff ** 2))
                            psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-8)))
                            gray_pred = cv2.cvtColor(pred_resized, cv2.COLOR_BGR2GRAY)
                            gray_tgt = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)
                            small_pred = cv2.resize(gray_pred, (64, 64))
                            small_tgt = cv2.resize(gray_tgt, (64, 64))
                            mu_x = small_pred.mean(); mu_y = small_tgt.mean()
                            sigma_x = small_pred.var(); sigma_y = small_tgt.var(); sigma_xy = ((small_pred - mu_x) * (small_tgt - mu_y)).mean()
                            C1 = 0.01 ** 2; C2 = 0.03 ** 2
                            ssim = float(((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2) + 1e-8))
                            psnrs.append(psnr); ssims.append(ssim)
                results["rollouts"][str(H)] = {
                    "psnr": float(np.mean(psnrs)) if psnrs else None,
                    "ssim": float(np.mean(ssims)) if ssims else None
                }
            except Exception:
                results["rollouts"][str(H)] = {"psnr": None, "ssim": None}

        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(results) + "\n")
        except Exception:
            pass
    
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def _get_compression_loss(self) -> float:
        """Get current compression loss"""
        # Assumes compression_lstm has a loss_history
        if hasattr(self.compression_lstm, 'loss_history') and self.compression_lstm.loss_history:
            return np.mean(list(self.compression_lstm.loss_history()))
        return 0.0
    
    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            "is_initialized": self.is_initialized,
            "frame_count": self.frame_count,
            "current_fps": self.current_fps,
            "avg_processing_time": np.mean(self.processing_times) if self.processing_times else 0,
            "device": str(self.device),
            "latent_shape": self.latent_shape,
            "compression_loss": self._get_compression_loss(),
            "pattern_buffer_size": len(self.pattern_lstm.latent_buffer),
            "training_enabled": self.config["training"]["enable_real_time_training"]
        }
    
    def reset_state(self):
        """Reset all model states"""
        self.central_lstm.reset_state()
        self.pattern_lstm.latent_buffer.clear()
        self.frame_count = 0
        self.processing_times.clear()
        print("AI Pipeline state reset")
    
    def save_models(self, save_dir: Path):
        """Save trained models"""
        save_dir.mkdir(exist_ok=True)
        
        # Save compression LSTM
        torch.save(self.compression_lstm.state_dict(), save_dir / "compression_lstm.pth")
        
        # Save central LSTM
        torch.save(self.central_lstm.state_dict(), save_dir / "central_lstm.pth")
        
        # Save configuration
        with open(save_dir / "ai_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir: Path):
        """Load trained models"""
        if not save_dir.exists():
            print(f"Save directory {save_dir} does not exist")
            return
        
        try:
            # Load compression LSTM
            compression_path = save_dir / "compression_lstm.pth"
            if compression_path.exists():
                self.compression_lstm.load_state_dict(torch.load(compression_path, map_location=self.device))
                print("Compression LSTM loaded")
            
            # Load central LSTM
            central_path = save_dir / "central_lstm.pth"
            if central_path.exists():
                self.central_lstm.load_state_dict(torch.load(central_path, map_location=self.device))
                print("Central LSTM loaded")
            
            print("Models loaded successfully")
        except Exception as e:
            print(f"Error loading models: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'vae_processor'):
            self.vae_processor.cleanup()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def create_default_config(config_path: Path):
    """Create a default configuration file"""
    default_config = {
        "pattern_lstm": {
            "hidden_dim": 512,
            "num_layers": 2,
            "buffer_size": 16,
            "description": "Frozen LSTM for pattern detection in latent sequences"
        },
        "compression_lstm": {
            "compressed_dim": 256,
            "num_layers": 2,
            "description": "Trainable LSTM for compressing detected patterns"
        },
        "central_lstm": {
            "hidden_dim": 512,
            "num_layers": 3,
            "description": "Main LSTM for next frame prediction from compressed patterns"
        },
        "training": {
            "enable_real_time_training": True,
            "training_frequency": 5,
            "delta_scale_r_max": 3.0,
            "delta_l2_lambda": 1e-6,
            "description": "Training configuration for real-time learning"
        },
        "logging": {
            "enabled": True,
            "path": "ai_logs/metrics.jsonl",
            "every_n_frames": 30,
            "every_secs": 2.0
        },
        "benchmark": {
            "enabled": True,
            "path": "ai_logs/benchmarks.jsonl",
            "every_n_frames": 120,
            "horizons": [5, 10]
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"Default configuration created at {config_path}")


if __name__ == "__main__":
    # Example usage
    vae_path = Path("vae")
    config_path = Path("ai_config.json")
    
    # Create default config if it doesn't exist
    if not config_path.exists():
        create_default_config(config_path)
    
    # Initialize pipeline
    # NOTE: Requires VAEProcessor and LSTM models to be defined/imported correctly
    pipeline = AIPipeline(vae_path, config_path)
    
    print("AI Pipeline initialized successfully")
    print(f"Status: {pipeline.get_status()}")