"""
Reward model implementations for RLHF and HCRL.

Classes
-------
_MLPBase          — shared 2-layer MLP + Adam optimizer (internal base class)
RewardModel       — preference-based reward model  [RLHF §2.2.3]
HCRLRewardModel   — scalar regression reward model [HCRL §III-A TAMER]
EnsembleRewardModel — ensemble with uncertainty queries [RLHF §2.2 modifications]

Functions
---------
oracle_preference — Boltzmann-rational simulated oracle [RLHF §2.2.2]

References
----------
[RLHF] Christiano et al., "Deep Reinforcement Learning from Human Preferences",
       NeurIPS 2017
[HCRL] Knox & Stone, "Interactively Shaping Agents via Human Reinforcement
       (TAMER)", K-CAP 2009
"""

from __future__ import annotations

import pathlib

import numpy as np

from cartpole import config as cfg


# ---------------------------------------------------------------------------
# Shared MLP base  (avoids duplicating forward / backward / Adam across models)
# ---------------------------------------------------------------------------

class _MLPBase:
    """
    Two-layer fully-connected network with tanh activations.

    Architecture:  Linear(obs_dim, hidden) → tanh
                → Linear(hidden, hidden)   → tanh
                → Linear(hidden, 1)

    Subclasses implement the specific loss function and training method.
    Adam optimizer state is shared here so both subclasses inherit it.
    """

    W1: np.ndarray; b1: np.ndarray
    W2: np.ndarray; b2: np.ndarray
    W3: np.ndarray; b3: np.ndarray
    lr: float

    # Adam hyper-parameters (fixed per the papers)
    _BETA1 = 0.9
    _BETA2 = 0.999
    _EPS   = 1e-8

    def _init_weights(self, obs_dim: int, hidden_dim: int, rng: np.random.Generator) -> None:
        s1 = np.sqrt(2.0 / obs_dim)
        s2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = rng.standard_normal((obs_dim, hidden_dim)) * s1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, hidden_dim)) * s2
        self.b2 = np.zeros(hidden_dim)
        self.W3 = rng.standard_normal((hidden_dim, 1)) * s2
        self.b3 = np.zeros(1)

    def _init_adam(self) -> None:
        self._t = 0
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]

    def _params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    # ------------------------------------------------------------------
    # Forward / backward pass
    # ------------------------------------------------------------------

    def _forward(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """obs: (N, obs_dim) → r: (N, 1), cache."""
        h1 = np.tanh(obs @ self.W1 + self.b1)
        h2 = np.tanh(h1  @ self.W2 + self.b2)
        r  = h2 @ self.W3 + self.b3
        return r, (obs, h1, h2)

    def _backward(
        self,
        d_r: np.ndarray,
        cache: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> list[np.ndarray]:
        obs, h1, h2 = cache
        d_W3 = h2.T @ d_r;          d_b3 = d_r.sum(0)
        d_h2 = d_r @ self.W3.T
        d_h2p = d_h2 * (1 - h2**2); d_W2 = h1.T @ d_h2p;  d_b2 = d_h2p.sum(0)
        d_h1  = d_h2p @ self.W2.T
        d_h1p = d_h1 * (1 - h1**2); d_W1 = obs.T @ d_h1p; d_b1 = d_h1p.sum(0)
        return [d_W1, d_b1, d_W2, d_b2, d_W3, d_b3]

    def _adam_step(self, grads: list[np.ndarray]) -> None:
        self._t += 1
        for i, (p, g) in enumerate(zip(self._params(), grads)):
            self._m[i] = self._BETA1 * self._m[i] + (1 - self._BETA1) * g
            self._v[i] = self._BETA2 * self._v[i] + (1 - self._BETA2) * g**2
            m_hat = self._m[i] / (1 - self._BETA1 ** self._t)
            v_hat = self._v[i] / (1 - self._BETA2 ** self._t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self._EPS)

    def predict(self, obs: np.ndarray) -> float | np.ndarray:
        scalar = obs.ndim == 1
        if scalar:
            obs = obs[None]
        r, _ = self._forward(obs)
        return float(r[0, 0]) if scalar else r[:, 0]

    # ------------------------------------------------------------------
    # Shared save / load helpers
    # ------------------------------------------------------------------

    def _save_weights(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3,
            lr=np.array(self.lr), adam_t=np.array(self._t),
            **{f"adam_m{i}": self._m[i] for i in range(6)},
            **{f"adam_v{i}": self._v[i] for i in range(6)},
        )

    def _load_weights(self, data: np.lib.npyio.NpzFile) -> None:
        for k in ("W1", "b1", "W2", "b2", "W3", "b3"):
            setattr(self, k, data[k])
        self.lr  = float(data["lr"])
        self._t  = int(data["adam_t"])
        self._m  = [data[f"adam_m{i}"] for i in range(6)]
        self._v  = [data[f"adam_v{i}"] for i in range(6)]


# ---------------------------------------------------------------------------
# Preference-based reward model  [RLHF §2.2.3 — Christiano et al. 2017]
# ---------------------------------------------------------------------------

class RewardModel(_MLPBase):
    """
    MLP trained with cross-entropy on pairwise segment preferences.

    Loss (Christiano et al. eq. 1 — Bradley-Terry model):
      P̂[σ¹ ≻ σ²] = exp(Σ r̂(oᵢ¹,aᵢ¹)) / (exp(Σ r̂(σ¹)) + exp(Σ r̂(σ²)))
      loss(r̂) = -Σ [μ·log P̂[σ¹≻σ²] + (1-μ)·log P̂[σ²≻σ¹]]
    """

    def __init__(
        self,
        obs_dim:    int   = 4,
        hidden_dim: int   = cfg.RLHF_RM_HIDDEN,
        lr:         float = cfg.RLHF_RM_LR,
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng()
        self.lr = lr
        self._init_weights(obs_dim, hidden_dim, rng)
        self._init_adam()

    def _preference_loss_and_grad(
        self, seg_a: np.ndarray, seg_b: np.ndarray, mu: float
    ) -> tuple[float, list[np.ndarray]]:
        r_a, cache_a = self._forward(seg_a)
        r_b, cache_b = self._forward(seg_b)
        sum_a, sum_b = float(r_a.sum()), float(r_b.sum())

        shift = max(sum_a, sum_b)
        exp_a = np.exp(sum_a - shift); exp_b = np.exp(sum_b - shift)
        z = exp_a + exp_b
        p_a = exp_a / z; p_b = exp_b / z

        loss = -(mu * np.log(p_a + 1e-8) + (1 - mu) * np.log(p_b + 1e-8))
        d_r_a = np.full_like(r_a, p_a - mu)
        d_r_b = np.full_like(r_b, p_b - (1 - mu))
        grads = [ga + gb for ga, gb in zip(self._backward(d_r_a, cache_a),
                                           self._backward(d_r_b, cache_b))]
        return float(loss), grads

    def train_on_preferences(
        self,
        segments_a: list[np.ndarray],
        segments_b: list[np.ndarray],
        preferences: list[float],
    ) -> float:
        """One gradient step on a batch of (seg_a, seg_b, μ) triples."""
        batch_grads = [np.zeros_like(p) for p in self._params()]
        total_loss  = 0.0
        for seg_a, seg_b, mu in zip(segments_a, segments_b, preferences):
            loss, grads = self._preference_loss_and_grad(seg_a, seg_b, mu)
            total_loss += loss
            for i, g in enumerate(grads):
                batch_grads[i] += g
        n = len(preferences)
        self._adam_step([g / n for g in batch_grads])
        return total_loss / n

    def save(self, file_path: str | pathlib.Path) -> None:
        path = pathlib.Path(file_path)
        self._save_weights(path)
        print(f"Reward model saved to {path}")

    @classmethod
    def load(cls, file_path: str | pathlib.Path) -> "RewardModel":
        data  = np.load(file_path)
        model = cls.__new__(cls)
        model._load_weights(data)
        print(f"Reward model loaded from {file_path}")
        return model


# ---------------------------------------------------------------------------
# Scalar-feedback regression model  [HCRL §III-A — TAMER / Knox & Stone 2009]
# ---------------------------------------------------------------------------

class HCRLRewardModel(_MLPBase):
    """
    MLP trained with MSE to regress scalar human feedback onto observations.

    During HCRL the oracle gives sparse ±signals. This model generalises those
    signals to all states, filling the silence gaps between oracle firings.

    Loss:  MSE(r̂(s), h)  where h is the oracle's human reward.
    """

    def __init__(
        self,
        obs_dim:    int   = 4,
        hidden_dim: int   = cfg.HCRL_RM_HIDDEN,
        lr:         float = cfg.HCRL_RM_LR,
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng()
        self.lr = lr
        self._init_weights(obs_dim, hidden_dim, rng)
        self._init_adam()

    def train_on_feedback(
        self,
        observations: np.ndarray,
        rewards:      np.ndarray,
        epochs:       int = cfg.HCRL_RM_EPOCHS,
    ) -> float:
        """Train on a batch of (observation, human_reward) pairs for `epochs` steps."""
        targets = rewards[:, None]
        loss = 0.0
        for _ in range(epochs):
            r_hat, cache = self._forward(observations)
            diff  = r_hat - targets
            loss  = float(np.mean(diff**2))
            d_r   = (2.0 / len(observations)) * diff
            self._adam_step(self._backward(d_r, cache))
        return loss

    def save(self, file_path: str | pathlib.Path) -> None:
        path = pathlib.Path(file_path)
        self._save_weights(path)
        print(f"HCRL reward model saved to {path}")

    @classmethod
    def load(cls, file_path: str | pathlib.Path) -> "HCRLRewardModel":
        data  = np.load(file_path)
        model = cls.__new__(cls)
        model._load_weights(data)
        print(f"HCRL reward model loaded from {file_path}")
        return model


# ---------------------------------------------------------------------------
# Simulated oracle  [RLHF §2.2.2 — Boltzmann rational model]
# ---------------------------------------------------------------------------

def oracle_preference(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    rng:   np.random.Generator,
    temperature: float = cfg.ORACLE_TEMPERATURE,
    error_prob:  float = cfg.ENSEMBLE_ERROR_PROB,
) -> float:
    """
    Simulated human oracle comparing two trajectory segments.

    Implements the Boltzmann-rational model from Christiano et al. §2.2.2:
      P(prefer A) = exp(score_A / T) / (exp(score_A / T) + exp(score_B / T))

    Also models the constant human-error rate from §2.2.3:
      With probability `error_prob`, the oracle responds uniformly at random.
      This prevents the probability of error from vanishing even when the
      reward difference is very large.

    Parameters
    ----------
    temperature : lower → more deterministic oracle
    error_prob  : probability of random response (human error)

    Returns
    -------
    1.0 (prefer A) or 0.0 (prefer B)
    """
    def _score(seg: np.ndarray) -> float:
        angle_stab = np.maximum(0.0, 1.0 - np.abs(seg[:, 2]) / cfg.CARTPOLE_ANGLE_LIMIT)
        pos_stab   = np.maximum(0.0, 1.0 - np.abs(seg[:, 0]) / cfg.CARTPOLE_POSITION_LIMIT)
        return float(np.mean(cfg.ORACLE_ANGLE_WEIGHT * angle_stab
                             + cfg.ORACLE_POSITION_WEIGHT * pos_stab))

    # Human-error model (§2.2.3): constant-rate random responses
    if rng.random() < error_prob:
        return 1.0 if rng.random() < 0.5 else 0.0

    score_a, score_b = _score(seg_a), _score(seg_b)
    T = temperature + 1e-8
    shift = max(score_a, score_b) / T
    exp_a = np.exp(score_a / T - shift)
    exp_b = np.exp(score_b / T - shift)
    p_a   = exp_a / (exp_a + exp_b)
    return 1.0 if rng.random() < p_a else 0.0


# ---------------------------------------------------------------------------
# Ensemble reward model  [RLHF §2.2 modifications — Christiano et al. 2017]
# ---------------------------------------------------------------------------

class EnsembleRewardModel:
    """
    Ensemble of K RewardModel predictors for full-paper RLHF.

    Implements three §2.2 improvements omitted from the basic RewardModel:

    1. **Ensemble** (bullet 1): K independent predictors trained on bootstrapped
       subsets of D; r̂ = mean of independently normalised predictors.

    2. **Uncertainty-based query selection** (§2.2.4): sample candidate pairs,
       select those with highest variance of (score_A − score_B) across members.

    3. **Reward normalisation** (§2.2.1): Welford online algorithm tracks running
       mean/std; normalise predicted rewards before passing to the RL agent.
    """

    def __init__(
        self,
        n_models:   int   = cfg.ENSEMBLE_N_MODELS,
        obs_dim:    int   = 4,
        hidden_dim: int   = cfg.RLHF_RM_HIDDEN,
        lr:         float = cfg.RLHF_RM_LR,
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng()
        self.n_models = n_models
        self.models: list[RewardModel] = [
            RewardModel(
                obs_dim=obs_dim, hidden_dim=hidden_dim, lr=lr,
                rng=np.random.default_rng(rng.integers(0, 2**31)),
            )
            for _ in range(n_models)
        ]
        self._reward_mean:  float = 0.0
        self._reward_var:   float = 1.0
        self._reward_count: int   = 0

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _raw_predictions(self, obs: np.ndarray) -> np.ndarray:
        """Per-model raw predictions. Shape: (n_models,)."""
        if obs.ndim == 1:
            obs = obs[None]
        return np.array([float(m.predict(obs)[0]) for m in self.models])

    def predict(self, obs: np.ndarray) -> float:
        """Mean prediction across ensemble (un-normalised)."""
        return float(np.mean(self._raw_predictions(obs)))

    def predict_with_variance(self, obs: np.ndarray) -> tuple[float, float]:
        """Return (mean, variance) across ensemble — variance = epistemic uncertainty."""
        raw = self._raw_predictions(obs)
        return float(np.mean(raw)), float(np.var(raw))

    def predict_normalised(self, obs: np.ndarray) -> float:
        """
        Predict and normalise to ~zero mean / unit std  [RLHF §2.2.1].

        Welford's online algorithm updates running statistics incrementally
        so normalisation improves as training progresses.
        """
        raw = self.predict(obs)
        self._reward_count += 1
        delta = raw - self._reward_mean
        self._reward_mean += delta / self._reward_count
        self._reward_var  += (delta * (raw - self._reward_mean) - self._reward_var) / self._reward_count
        return (raw - self._reward_mean) / max(np.sqrt(self._reward_var), 1e-8)

    def segment_score(self, segment: np.ndarray) -> np.ndarray:
        """Summed reward over a trajectory segment per ensemble member. Shape: (n_models,)."""
        return np.array([float(m.predict(segment).sum()) for m in self.models])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on_preferences(
        self,
        segments_a:  list[np.ndarray],
        segments_b:  list[np.ndarray],
        preferences: list[float],
        validation_fraction: float = cfg.ENSEMBLE_VAL_FRACTION,
    ) -> float:
        """
        Train each member on a bootstrapped subset of D  [RLHF §2.2 bullet 1-2].

        Each model is trained on |D| triples sampled with replacement;
        `validation_fraction` of the sample is held out for validation.
        """
        n   = len(preferences)
        rng = np.random.default_rng()
        total_loss = 0.0

        for model in self.models:
            idx  = rng.integers(0, n, size=n)
            val  = max(1, int(n * validation_fraction))
            tidx = idx[val:] if len(idx[val:]) > 0 else idx
            loss = model.train_on_preferences(
                [segments_a[i] for i in tidx],
                [segments_b[i] for i in tidx],
                [preferences[i] for i in tidx],
            )
            total_loss += loss

        return total_loss / self.n_models

    # ------------------------------------------------------------------
    # Uncertainty-based query selection  [RLHF §2.2.4]
    # ------------------------------------------------------------------

    def select_uncertain_pairs(
        self,
        segment_buffer: list[np.ndarray],
        n_pairs:        int,
        n_candidates:   int | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, int]]]:
        """
        Select the n_pairs segment pairs with highest ensemble disagreement.

        1. Sample n_candidates random pairs from the buffer.
        2. Compute variance of (score_A − score_B) across ensemble members.
        3. Return top n_pairs by variance (most uncertain first).
        """
        rng = rng or np.random.default_rng()
        n_candidates = n_candidates or max(n_pairs * cfg.ENSEMBLE_CANDIDATES_MULT, 50)
        buf = len(segment_buffer)

        candidates = [(int(i), int(j))
                      for i, j in (rng.choice(buf, size=2, replace=False)
                                   for _ in range(n_candidates))]

        uncertainties = np.array([
            float(np.var(self.segment_score(segment_buffer[i])
                         - self.segment_score(segment_buffer[j])))
            for i, j in candidates
        ])

        top = np.argsort(uncertainties)[::-1][:n_pairs]
        segs_a, segs_b, chosen = [], [], []
        for k in top:
            i, j = candidates[k]
            segs_a.append(segment_buffer[i])
            segs_b.append(segment_buffer[j])
            chosen.append((i, j))
        return segs_a, segs_b, chosen

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dir_path: str | pathlib.Path, prefix: str = "ensemble") -> None:
        path = pathlib.Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        for k, model in enumerate(self.models):
            model.save(path / f"{prefix}_model_{k}.npz")
        np.savez(path / f"{prefix}_normalizer.npz",
                 reward_mean=np.array(self._reward_mean),
                 reward_var=np.array(self._reward_var),
                 reward_count=np.array(self._reward_count))
        print(f"Ensemble ({self.n_models} models) saved to {path}/")

    @classmethod
    def load(cls, dir_path: str | pathlib.Path, prefix: str = "ensemble") -> "EnsembleRewardModel":
        path = pathlib.Path(dir_path)
        k = 0
        while (path / f"{prefix}_model_{k}.npz").exists():
            k += 1
        if k == 0:
            raise FileNotFoundError(f"No ensemble models found in {path}")

        obj = cls.__new__(cls)
        obj.n_models = k
        obj.models   = [RewardModel.load(path / f"{prefix}_model_{i}.npz") for i in range(k)]

        norm = path / f"{prefix}_normalizer.npz"
        if norm.exists():
            data = np.load(norm)
            obj._reward_mean  = float(data["reward_mean"])
            obj._reward_var   = float(data["reward_var"])
            obj._reward_count = int(data["reward_count"])
        else:
            obj._reward_mean = 0.0; obj._reward_var = 1.0; obj._reward_count = 0

        print(f"Ensemble ({k} models) loaded from {path}/")
        return obj
