# OLM Pipeline - Frozen VAE Latent Dynamics

**🚀 MAJOR MILESTONE: First functioning model on the OLM pipeline - A major stepping stone towards completing OLM**

A real-time AI pipeline that processes camera frames through a frozen VAE and three-stage LSTM system for video state extraction and next-frame prediction. This represents the first working implementation of the OLM (Object-Level Manipulation) pipeline architecture.

## Architecture Overview

![OLM Pipeline Architecture](images/architecture_diagram.png)

This system implements a novel approach to video understanding using:

1. **Frozen VAE Encoder/Decoder**: Stable Diffusion VAE for consistent latent space representation
2. **Pattern LSTM**: Frozen LSTM that aggregates short latent sequences for temporal pattern detection
3. **Compression LSTM**: Trainable LSTM that compresses patterns while avoiding variance collapse
4. **Central LSTM**: Main prediction network that forecasts Δz (latent deltas) for next-frame generation

### Key Innovation: Clean Separation of Concerns
- **Sensing**: Frozen VAE provides stable manifold
- **Temporal Aggregation**: Pattern/Compression LSTMs handle sequence processing
- **Prediction**: Central LSTM learns dynamics on compressed representations

This separation makes the system transparent, debuggable, and faster to iterate compared to end-to-end approaches.

## Performance Results

### Real-time Metrics Dashboard
![Real-time Metrics](images/metrics_dashboard.png)

The system demonstrates stable performance across key metrics:
- **One-step prediction**: PSNR ~25-26 dB, SSIM ~0.99, latent MSE ~2.6e-3–3.5e-3
- **Causality verification**: cos(ẑₜ₊₁, zₜ₊₁) > cos(zₜ, zₜ₊₁) consistently maintained
- **Processing rate**: Stable 24 FPS with real-time training enabled
- **Training stability**: Loss convergence with gradient clipping, no variance collapse

### Open-Loop Rollout Performance
![Benchmark Rollouts](images/benchmark_rollouts.png)

**Multi-step prediction capabilities** (no teacher forcing):
- **5-step rollouts**: PSNR ~17.6–18.4 dB, SSIM ~0.94–0.95
- **10-step rollouts**: PSNR ~16.1–17.0 dB, SSIM ~0.93–0.94
- **Stability**: Monotonic error growth, no divergence or blow-ups
- **Compressor health**: Per-dimension variance slowly increases, avoiding collapse

### Why These Results Matter
1. **Validates OLM approach**: Proves that structured state extraction works for video prediction
2. **Short-horizon stability**: Demonstrates reliable forecasting without end-to-end VAE training
3. **Debuggable pipeline**: Each component can be analyzed and verified independently
4. **Foundation for editing**: Stable latent dynamics enable identity-conditioned modifications

## Features

- Real-time camera capture with configurable settings
- VAE-based latent extraction at 24fps
- Three-stage LSTM pipeline for pattern learning
- Real-time training of compression and prediction models
- Split-view GUI showing camera feed and AI predictions
- Performance metrics and status monitoring
- Configurable AI pipeline parameters
- Comprehensive telemetry system (PSNR/SSIM, latent MSE, cosine similarities, rollout analysis)

## OLM Project Roadmap

### ✅ **Completed: Foundation Pipeline**
- Frozen VAE + three-stage LSTM architecture
- Stable video state extraction and prediction
- Real-time training with scheduled sampling
- Comprehensive metrics and benchmarking system

### 🎯 **Next: Identity-Conditioned Background Removal**
- **Background-plate LoRA**: Decoder-side LoRA (rank 4-8) conditioned on identity token
- **Training**: Supervised by clean background frames to map "current latent" → "clean background latent"
- **Inference**: Toggle identity flag to remove subject ("world without me")
- **Use cases**: Privacy-preserving streaming, AR compositing, clean data collection

### 🔮 **Future: Advanced Identity Manipulation**
- **ICMI**: Identity-conditioned matting + latent inpainting for moving cameras
- **Subject replacement**: Target identity LoRA with pose-driven keypoints
- **Temporal consistency**: Flow-based loss to avoid jitter
- **Live adaptation**: Integration with novelty driver and VAE fine-tuning

### 🛡️ **Ethical Framework**
- Consent requirements for identity replacement
- Clear labeling for edited outputs
- Localized LoRA modifications to minimize background drift

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the VAE model:
   - The VAE model files are required but not included due to size limits
   - Download `diffusion_pytorch_model.safetensors` from [Hugging Face Stable Diffusion VAE](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main)
   - Place it in the `vae/` folder alongside the included `config.json`

3. Verify VAE setup:
   - `vae/config.json` ✓ (included)
   - `vae/diffusion_pytorch_model.safetensors` (download required)

## Usage

### Basic Camera GUI
```bash
python camera_gui.py
```

This launches the GUI with:
- Camera feed display
- Start/Stop controls
- AI prediction display (if enabled)
- Performance metrics

### AI Pipeline Configuration

The system uses several configuration files:

#### `camera_config.json`
- Camera settings (index, resolution, mirror mode)
- AI pipeline enable/disable
- VAE model path

#### `ai_config.json` (auto-generated)
- LSTM model parameters
- Training configuration
- Pattern buffer settings

### Key Controls

- **Start**: Begin camera capture and AI processing
- **Stop**: Stop camera capture
- **Reset AI**: Reset AI pipeline state
- **Quit**: Exit application

## Architecture

### VAE Processor (`vae_processor.py`)
- Loads Stable Diffusion VAE model
- Processes frames at 24fps rate limiting
- Encodes frames to latents
- Decodes predicted latents back to frames

### LSTM Models (`lstm_models.py`)

#### Pattern LSTM
- Frozen LSTM for pattern detection
- Maintains rolling buffer of latents
- Outputs hidden states representing patterns

#### Compression LSTM
- Trainable LSTM that compresses patterns
- Real-time training with reconstruction loss
- Outputs compressed representations

#### Central LSTM
- Main prediction LSTM
- Takes compressed patterns as input
- Predicts next frame latents

### AI Pipeline (`AI.py`)
- Orchestrates the entire processing pipeline
- Manages model initialization and state
- Provides performance metrics and monitoring
- Handles model saving/loading

### GUI Integration (`camera_gui.py`)
- Split-view display of camera and AI predictions
- Real-time status and metrics display
- AI pipeline controls and monitoring

## Configuration

### Camera Settings
```json
{
  "camera_index": 0,
  "frame_width": 1280,
  "frame_height": 720,
  "mirror_preview": true,
  "enable_ai": true
}
```

### AI Pipeline Settings
```json
{
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
  }
}
```

## Performance

The system is designed for real-time operation:
- Camera capture: ~30-60 FPS (configurable)
- AI processing: 24 FPS (rate limited)
- VAE encoding/decoding: GPU accelerated
- LSTM inference: CPU/GPU depending on hardware

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Webcam or video source
- Stable Diffusion VAE model files

## Troubleshooting

### Common Issues

1. **Camera not detected**: Check `camera_index` in config
2. **VAE loading errors**: Ensure model files are in `vae/` folder
3. **Performance issues**: Reduce resolution or disable AI
4. **Memory errors**: Lower batch sizes or use CPU mode

### Performance Optimization

- Use GPU acceleration when available
- Adjust frame resolution based on hardware
- Monitor memory usage during long sessions
- Save trained models periodically

## Future Enhancements

- Multi-camera support
- Advanced pattern recognition algorithms
- Temporal consistency improvements
- Real-time model fine-tuning
- Export capabilities for generated sequences

## License

This project is for research and educational purposes.
