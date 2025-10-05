from __future__ import annotations
import torch
import math
import time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List, Union


@dataclass
class CacheEntry:
    c: torch.Tensor               # cached compressed vector [B, D]
    best_depth: int               # chosen K*
    score_ema: float              # quality score EMA
    cost_ema: float               # ms EMA
    motion_ema: float             # avg motion norm
    ts: float                     # last update time
    ver: int = 1                  # normalization/proj version
    alt_depths: Optional[Dict[int, float]] = None  # optional depth->score


class LSHCache:
    """
    Sign-LSH over a fixed random projection of latent vectors.
    Provides: hash(z), insert, lookup with Hamming radius (r in {0,1}).
    
    Optimized for speed using vectorized hash generation.
    """
    def __init__(self, in_dim: int, n_bits: int = 128, seed: int = 13, device: Union[str, torch.device] = 'cpu'):
        self.device = torch.device(device)
        
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        
        # R, mu, and sigma are now initialized on the target device for faster access
        self.R = torch.randn(in_dim, n_bits, generator=g, device=self.device)  # random projection [D, n_bits]
        self.mu = torch.zeros(in_dim, device=self.device)                      # online mean
        self.sigma = torch.ones(in_dim, device=self.device)                    # online std dev
        
        self.frozen_norm = False
        self.store: Dict[int, CacheEntry] = {}
        self.n_bits = n_bits
        self.version = 1

        # Precompute neighbors (bit masks) and weights for vectorized hash generation
        self._bit_masks = [1 << i for i in range(n_bits)]
        
        # Precompute powers of 2 for vectorized bit packing (moved to device)
        # Weights for [2^0, 2^1, 2^2, ..., 2^(n_bits-1)]
        self._hash_weights = torch.pow(2, torch.arange(n_bits, device=self.device)).to(torch.int64)

    def update_norm(self, x: torch.Tensor):
        """Online update of mean and standard deviation."""
        if self.frozen_norm:
            return
        # Ensure x is on the correct device for moment calculation
        x = x.to(self.device)
        with torch.no_grad():
            # Handle potential batch dimension
            mu = x.mean(dim=0)
            sd = x.std(dim=0, unbiased=False)
            
            # EMA updates
            self.mu = 0.99 * self.mu + 0.01 * mu
            self.sigma = 0.99 * self.sigma + 0.01 * torch.clamp(sd, min=1e-3)

    def freeze_norm(self):  # call after warmup ~ few hundred frames
        """Freezes the normalization parameters and increments the version."""
        self.frozen_norm = True
        self.version += 1
        print(f"LSH Cache norm frozen. New version: {self.version}. Cache size: {len(self.store)}")

    def _sign_hash_vectorized(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized sign hash calculation. Supports batch input: [B, D] or single input: [D].
        Returns: (hash_tensor [B], normalized_z [B, D])
        """
        # Ensure input z is at least 2D [B, D] for consistent matrix operations
        if z.ndim == 1:
            z = z.unsqueeze(0)  # [D] -> [1, D]

        # Normalize: zn = (z - mu) / sigma
        zn = (z - self.mu) / self.sigma

        # Project: proj = zn @ R. Result is [B, n_bits]
        proj = zn @ self.R

        # Sign bits: bits [B, n_bits] (0/1)
        bits = (proj >= 0).to(torch.int64)
        
        # Bit packing (vectorized dot product): [B, n_bits] @ [n_bits] -> [B]
        # This replaces the slow Python for loop
        h_tensor = bits @ self._hash_weights
        
        return h_tensor, zn.squeeze(0) if h_tensor.numel() == 1 else zn

    def hash(self, z: torch.Tensor) -> Tuple[int, torch.Tensor]:
        """Public API for single-vector hashing, returning hash int and normalized vector."""
        h_tensor, zn = self._sign_hash_vectorized(z)
        return int(h_tensor.item()), zn

    def hamming(self, a: int, b: int) -> int:
        """Calculates the Hamming distance between two hash integers."""
        return int((a ^ b).bit_count())

    def neighbors_r1(self, h: int) -> List[int]:
        """Returns all 1-bit Hamming distance neighbors for a hash h."""
        return [h ^ m for m in self._bit_masks]

    def lookup(self, z: torch.Tensor, radius: int = 1) -> Tuple[Optional[CacheEntry], int, int]:
        """
        Returns (entry, h, dist). Searches r=0 then r=1.
        """
        h, _zn = self.hash(z) # Use public hash method
        
        # r=0: Direct hit
        if h in self.store:
            return self.store[h], h, 0
            
        # r=1: Hamming distance 1 search
        if radius >= 1:
            for nh in self.neighbors_r1(h):
                e = self.store.get(nh)
                if e is not None:
                    # Found in neighbor bin, return neighbor hash (nh)
                    return e, nh, 1
                    
        return None, h, self.n_bits # Use max distance when no entry is found

    def insert(self, h: int, entry: CacheEntry):
        """Inserts a new CacheEntry into the store at hash h."""
        self.store[h] = entry

    def update_entry(self, h: int, *, score=None, cost=None, motion=None, best_depth=None):
        """Updates the EMA metrics and timestamp of an existing cache entry."""
        e = self.store.get(h)
        if e is None:
            return
        
        # EMA updates (alpha = 0.1)
        if score is not None:
            e.score_ema = 0.9 * e.score_ema + 0.1 * float(score)
        if cost is not None:
            e.cost_ema = 0.9 * e.cost_ema + 0.1 * float(cost)
        if motion is not None:
            e.motion_ema = 0.9 * e.motion_ema + 0.1 * float(motion)
            
        if best_depth is not None:
            e.best_depth = int(best_depth)
            
        e.ts = time.time()
        
    def prune_cache(self, max_age_seconds: float = 300, max_size: int = 10000) -> int:
        """
        Removes entries older than max_age_seconds and/or trims the cache to max_size 
        by removing the oldest entries.
        """
        current_time = time.time()
        keys_to_delete = []
        
        # 1. Prune by Age
        for h, entry in self.store.items():
            if (current_time - entry.ts) > max_age_seconds:
                keys_to_delete.append(h)
        
        for h in keys_to_delete:
            del self.store[h]
            
        initial_delete_count = len(keys_to_delete)
        
        # 2. Prune by Size (if still over limit)
        if len(self.store) > max_size:
            # Get all entries, sort by timestamp (oldest first)
            all_entries = sorted(self.store.items(), key=lambda item: item[1].ts)
            
            num_to_trim = len(self.store) - max_size
            for i in range(num_to_trim):
                h, _ = all_entries[i]
                del self.store[h]
                
            print(f"LSH Cache: Trimmed {num_to_trim} entries to maintain max size of {max_size}.")
            
        print(f"LSH Cache: Pruned {initial_delete_count} entries by age. Current size: {len(self.store)}")
        return initial_delete_count