from __future__ import annotations
import torch, math, time
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List


@dataclass
class CacheEntry:
    c: torch.Tensor                  # cached compressed vector [B, D]
    best_depth: int                  # chosen K*
    score_ema: float                 # quality score EMA
    cost_ema: float                  # ms EMA
    motion_ema: float                # avg motion norm
    ts: float                        # last update time
    ver: int = 1                     # normalization/proj version
    alt_depths: Optional[Dict[int, float]] = None  # optional depth->score


class LSHCache:
    """
    Sign-LSH over a fixed random projection of latent vectors.
    Provides: hash(z), insert, lookup with Hamming radius (r in {0,1,2}).
    """
    def __init__(self, in_dim: int, n_bits: int = 128, seed: int = 13):
        g = torch.Generator(device='cpu')
        g.manual_seed(seed)
        self.R = torch.randn(in_dim, n_bits, generator=g)  # random projection
        self.mu = torch.zeros(in_dim)
        self.sigma = torch.ones(in_dim)
        self.frozen_norm = False
        self.store: Dict[int, CacheEntry] = {}
        self.n_bits = n_bits
        self.version = 1

        # precompute neighbors for r=1 (flip masks)
        self._bit_masks = [1 << i for i in range(n_bits)]

    def update_norm(self, x: torch.Tensor):
        if self.frozen_norm:
            return
        # online moment update (small batch BxD)
        with torch.no_grad():
            mu = x.mean(dim=0).cpu()
            sd = x.std(dim=0, unbiased=False).cpu()
            self.mu = 0.99 * self.mu + 0.01 * mu
            self.sigma = 0.99 * self.sigma + 0.01 * torch.clamp(sd, min=1e-3)

    def freeze_norm(self):  # call after warmup ~ few hundred frames
        self.frozen_norm = True
        self.version += 1

    def _sign_hash(self, z: torch.Tensor) -> Tuple[int, torch.Tensor]:
        # z: [D], normalize -> project -> sign bits -> int hash
        zn = (z - self.mu.to(z.device)) / self.sigma.to(z.device)
        proj = zn @ self.R.to(z.device)          # [n_bits]
        bits = (proj >= 0).to(torch.int64)       # 0/1
        # pack bits to int
        h = 0
        for i in range(self.n_bits):
            h |= (int(bits[i].item()) << i)
        return h, zn

    def hamming(self, a: int, b: int) -> int:
        return int((a ^ b).bit_count())

    def neighbors_r1(self, h: int) -> List[int]:
        return [h ^ m for m in self._bit_masks]

    def lookup(self, z: torch.Tensor, radius: int = 1) -> Tuple[Optional[CacheEntry], int, int]:
        """
        Returns (entry, h, dist). Searches r=0 then r=1 (no r=2 for speed).
        """
        h, _zn = self._sign_hash(z)
        if h in self.store:
            return self.store[h], h, 0
        if radius >= 1:
            for nh in self.neighbors_r1(h):
                e = self.store.get(nh)
                if e is not None:
                    return e, nh, 1
        return None, h, 999

    def insert(self, h: int, entry: CacheEntry):
        self.store[h] = entry

    def update_entry(self, h: int, *, score=None, cost=None, motion=None, best_depth=None):
        e = self.store.get(h)
        if e is None:
            return
        if score is not None:
            e.score_ema = 0.9 * e.score_ema + 0.1 * float(score)
        if cost is not None:
            e.cost_ema = 0.9 * e.cost_ema + 0.1 * float(cost)
        if motion is not None:
            e.motion_ema = 0.9 * e.motion_ema + 0.1 * float(motion)
        if best_depth is not None:
            e.best_depth = int(best_depth)
        e.ts = time.time()


