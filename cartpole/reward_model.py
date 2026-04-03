"""
Preference-based reward model for RLHF (Christiano et al., 2017).

"Deep Reinforcement Learning from Human Preferences"
Christiano, Leike, Brown, Martic, Legg, Amodei (NeurIPS 2017)

The reward model is a small MLP trained to predict which of two trajectory
segments a human would prefer.  A simulated oracle (Boltzmann-rational model)
replaces real human annotators.
"""

import pathlib

import numpy as np


# ---------------------------------------------------------------------------
# Reward model (2-layer MLP, pure numpy + Adam)
# ---------------------------------------------------------------------------

class RewardModel:
    """
    Small fully-connected network: obs -> scalar reward prediction.

    Architecture:  Linear(obs_dim, hidden) -> tanh
                -> Linear(hidden, hidden)  -> tanh
                -> Linear(hidden, 1)

    Trained with cross-entropy on pairwise segment preferences
    (Christiano et al. eq. 1).
    """

    def __init__(
        self,
        obs_dim: int = 4,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng()

        scale1 = np.sqrt(2.0 / obs_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1: np.ndarray = rng.standard_normal((obs_dim, hidden_dim)) * scale1
        self.b1: np.ndarray = np.zeros(hidden_dim)
        self.W2: np.ndarray = rng.standard_normal((hidden_dim, hidden_dim)) * scale2
        self.b2: np.ndarray = np.zeros(hidden_dim)
        self.W3: np.ndarray = rng.standard_normal((hidden_dim, 1)) * scale2
        self.b3: np.ndarray = np.zeros(1)

        self.lr = lr
        self._init_adam()

    # ------------------------------------------------------------------
    # Adam optimiser state
    # ------------------------------------------------------------------

    def _init_adam(self) -> None:
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._eps = 1e-8
        self._t = 0
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]

    def _params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def _forward(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """obs: (T, obs_dim)  ->  r: (T, 1),  cache for backprop."""
        h1 = np.tanh(obs @ self.W1 + self.b1)   # (T, hidden)
        h2 = np.tanh(h1  @ self.W2 + self.b2)   # (T, hidden)
        r  = h2 @ self.W3 + self.b3              # (T, 1)
        return r, (obs, h1, h2)

    def predict(self, obs: np.ndarray) -> float | np.ndarray:
        """
        Predict reward for one or more observations.

        Parameters
        ----------
        obs : (obs_dim,) or (T, obs_dim)

        Returns
        -------
        float  if obs is 1-D,  np.ndarray (T,) otherwise.
        """
        scalar = obs.ndim == 1
        if scalar:
            obs = obs[None]
        r, _ = self._forward(obs)
        return float(r[0, 0]) if scalar else r[:, 0]

    # ------------------------------------------------------------------
    # Backprop
    # ------------------------------------------------------------------

    def _backward(
        self,
        d_r: np.ndarray,
        cache: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> list[np.ndarray]:
        """Compute gradients given upstream gradient d_r: (T, 1)."""
        obs, h1, h2 = cache

        d_W3 = h2.T @ d_r
        d_b3 = d_r.sum(axis=0)
        d_h2 = d_r @ self.W3.T

        d_h2_pre = d_h2 * (1.0 - h2 ** 2)
        d_W2 = h1.T @ d_h2_pre
        d_b2 = d_h2_pre.sum(axis=0)
        d_h1 = d_h2_pre @ self.W2.T

        d_h1_pre = d_h1 * (1.0 - h1 ** 2)
        d_W1 = obs.T @ d_h1_pre
        d_b1 = d_h1_pre.sum(axis=0)

        return [d_W1, d_b1, d_W2, d_b2, d_W3, d_b3]

    # ------------------------------------------------------------------
    # Preference loss (Christiano et al. eq. 1)
    # ------------------------------------------------------------------

    def _preference_loss_and_grad(
        self,
        seg_a: np.ndarray,
        seg_b: np.ndarray,
        mu: float,
    ) -> tuple[float, list[np.ndarray]]:
        """
        Compute cross-entropy preference loss and gradients for one pair.

        L = - [ mu * log P(A>B)  +  (1-mu) * log P(B>A) ]
        P(A>B) = exp(Σ r_hat(A)) / ( exp(Σ r_hat(A)) + exp(Σ r_hat(B)) )

        Parameters
        ----------
        seg_a, seg_b : (T, obs_dim)  trajectory segments
        mu           : 1.0 = prefer A,  0.0 = prefer B,  0.5 = tie
        """
        r_a, cache_a = self._forward(seg_a)   # (T_a, 1)
        r_b, cache_b = self._forward(seg_b)   # (T_b, 1)

        sum_a = float(r_a.sum())
        sum_b = float(r_b.sum())

        # Numerically stable softmax over {sum_a, sum_b}
        shift   = max(sum_a, sum_b)
        exp_a   = np.exp(sum_a - shift)
        exp_b   = np.exp(sum_b - shift)
        z       = exp_a + exp_b
        p_a     = exp_a / z          # P(prefer A)
        p_b     = exp_b / z          # P(prefer B)

        loss = -(mu * np.log(p_a + 1e-8) + (1.0 - mu) * np.log(p_b + 1e-8))

        # dL/d(sum_a) = p_a - mu        (derived from softmax cross-entropy)
        # dL/d(sum_b) = p_b - (1 - mu)
        # dL/d(r_t^a) = dL/d(sum_a)  for all t in A  (sum is linear)
        d_r_a = np.full_like(r_a, p_a - mu)
        d_r_b = np.full_like(r_b, p_b - (1.0 - mu))

        grads_a = self._backward(d_r_a, cache_a)
        grads_b = self._backward(d_r_b, cache_b)

        combined = [ga + gb for ga, gb in zip(grads_a, grads_b)]
        return float(loss), combined

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on_preferences(
        self,
        segments_a: list[np.ndarray],
        segments_b: list[np.ndarray],
        preferences: list[float],
    ) -> float:
        """
        One gradient step on a batch of (seg_a, seg_b, mu) triples.

        Returns average loss over the batch.
        """
        batch_grads = [np.zeros_like(p) for p in self._params()]
        total_loss = 0.0

        for seg_a, seg_b, mu in zip(segments_a, segments_b, preferences):
            loss, grads = self._preference_loss_and_grad(seg_a, seg_b, mu)
            total_loss += loss
            for i, g in enumerate(grads):
                batch_grads[i] += g

        n = len(preferences)
        batch_grads = [g / n for g in batch_grads]

        # Adam update
        self._t += 1
        params = self._params()
        for i, (p, g) in enumerate(zip(params, batch_grads)):
            self._m[i] = self._beta1 * self._m[i] + (1.0 - self._beta1) * g
            self._v[i] = self._beta2 * self._v[i] + (1.0 - self._beta2) * g ** 2
            m_hat = self._m[i] / (1.0 - self._beta1 ** self._t)
            v_hat = self._v[i] / (1.0 - self._beta2 ** self._t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self._eps)

        return total_loss / n

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, file_path: str | pathlib.Path) -> None:
        """
        Save all model weights and hyper-parameters to a .npz file.

        Parameters
        ----------
        file_path : path to the output file (e.g. 'experiment-result/reward_model.npz')
        """
        path = pathlib.Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
            lr=np.array(self.lr),
            adam_t=np.array(self._t),
            adam_m0=self._m[0], adam_m1=self._m[1],
            adam_m2=self._m[2], adam_m3=self._m[3],
            adam_m4=self._m[4], adam_m5=self._m[5],
            adam_v0=self._v[0], adam_v1=self._v[1],
            adam_v2=self._v[2], adam_v3=self._v[3],
            adam_v4=self._v[4], adam_v5=self._v[5],
        )
        print(f"Reward model saved to {path}")

    @classmethod
    def load(cls, file_path: str | pathlib.Path) -> "RewardModel":
        """
        Load a saved reward model from a .npz file.

        Parameters
        ----------
        file_path : path to the .npz file produced by :meth:`save`

        Returns
        -------
        RewardModel with weights and Adam state restored.
        """
        data = np.load(file_path)
        obs_dim, hidden_dim = data["W1"].shape

        model = cls.__new__(cls)
        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = data["b2"]
        model.W3 = data["W3"]
        model.b3 = data["b3"]
        model.lr = float(data["lr"])

        model._beta1 = 0.9
        model._beta2 = 0.999
        model._eps   = 1e-8
        model._t     = int(data["adam_t"])
        model._m     = [data[f"adam_m{i}"] for i in range(6)]
        model._v     = [data[f"adam_v{i}"] for i in range(6)]

        print(f"Reward model loaded from {file_path}")
        return model


# ---------------------------------------------------------------------------
# Simulated human oracle (Boltzmann rational model)
# ---------------------------------------------------------------------------

def oracle_preference(
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    rng: np.random.Generator,
    temperature: float = 0.05,
) -> float:
    """
    Simulated human oracle that compares two trajectory segments.

    Uses a Boltzmann rational model (Christiano et al. section 2.2):
      P(prefer A) = exp(score_A / T) / (exp(score_A / T) + exp(score_B / T))

    The "true" segment score is computed from pole angle and cart position
    stability — the same ground truth the CartPole env implicitly encodes.

    Parameters
    ----------
    seg_a, seg_b  : (T, obs_dim)  trajectory segments
    rng           : random generator for stochastic choice
    temperature   : lower = more deterministic oracle  (default 0.05)

    Returns
    -------
    float  :  1.0 (prefer A),  0.0 (prefer B)
    """
    def _score(seg: np.ndarray) -> float:
        pole_angles = np.abs(seg[:, 2])
        cart_xs     = np.abs(seg[:, 0])
        angle_stab  = np.maximum(0.0, 1.0 - pole_angles / 0.2095)
        pos_stab    = np.maximum(0.0, 1.0 - cart_xs    / 2.4)
        return float(np.mean(0.7 * angle_stab + 0.3 * pos_stab))

    score_a = _score(seg_a)
    score_b = _score(seg_b)

    # Numerically stable Boltzmann
    shift   = max(score_a, score_b) / (temperature + 1e-8)
    exp_a   = np.exp(score_a / (temperature + 1e-8) - shift)
    exp_b   = np.exp(score_b / (temperature + 1e-8) - shift)
    p_a     = exp_a / (exp_a + exp_b)

    return 1.0 if rng.random() < p_a else 0.0


# ---------------------------------------------------------------------------
# HCRL reward model — regression on direct human scalar feedback (TAMER-style)
# ---------------------------------------------------------------------------

class HCRLRewardModel:
    """
    Regression MLP that generalises sparse human feedback to unseen states.

    During HCRL training the human presses keys at a small fraction of
    timesteps.  This model is trained on those labelled (observation, reward)
    pairs and then predicts a reward for *every* state, filling the silence
    gaps that the raw TAMER signal leaves.

    Architecture:  Linear(obs_dim, hidden) -> tanh
                -> Linear(hidden, hidden)  -> tanh
                -> Linear(hidden, 1)

    Loss: MSE between predicted reward and human-given reward.
    Optimiser: Adam.
    """

    def __init__(
        self,
        obs_dim: int = 4,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng()

        scale1 = np.sqrt(2.0 / obs_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)

        self.W1: np.ndarray = rng.standard_normal((obs_dim, hidden_dim)) * scale1
        self.b1: np.ndarray = np.zeros(hidden_dim)
        self.W2: np.ndarray = rng.standard_normal((hidden_dim, hidden_dim)) * scale2
        self.b2: np.ndarray = np.zeros(hidden_dim)
        self.W3: np.ndarray = rng.standard_normal((hidden_dim, 1)) * scale2
        self.b3: np.ndarray = np.zeros(1)

        self.lr = lr
        self._init_adam()

    # ------------------------------------------------------------------
    # Adam optimiser state
    # ------------------------------------------------------------------

    def _init_adam(self) -> None:
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._eps = 1e-8
        self._t = 0
        self._m = [np.zeros_like(p) for p in self._params()]
        self._v = [np.zeros_like(p) for p in self._params()]

    def _params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def _forward(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """obs: (N, obs_dim)  ->  r_hat: (N, 1),  cache."""
        h1 = np.tanh(obs @ self.W1 + self.b1)
        h2 = np.tanh(h1  @ self.W2 + self.b2)
        r  = h2 @ self.W3 + self.b3
        return r, (obs, h1, h2)

    def predict(self, obs: np.ndarray) -> float | np.ndarray:
        """
        Predict human reward for one or a batch of observations.

        Parameters
        ----------
        obs : (obs_dim,) or (N, obs_dim)

        Returns
        -------
        float  if obs is 1-D,  np.ndarray (N,) otherwise.
        """
        scalar = obs.ndim == 1
        if scalar:
            obs = obs[None]
        r, _ = self._forward(obs)
        return float(r[0, 0]) if scalar else r[:, 0]

    def _backward(
        self,
        d_r: np.ndarray,
        cache: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> list[np.ndarray]:
        obs, h1, h2 = cache

        d_W3 = h2.T @ d_r
        d_b3 = d_r.sum(axis=0)
        d_h2 = d_r @ self.W3.T

        d_h2_pre = d_h2 * (1.0 - h2 ** 2)
        d_W2 = h1.T @ d_h2_pre
        d_b2 = d_h2_pre.sum(axis=0)
        d_h1 = d_h2_pre @ self.W2.T

        d_h1_pre = d_h1 * (1.0 - h1 ** 2)
        d_W1 = obs.T @ d_h1_pre
        d_b1 = d_h1_pre.sum(axis=0)

        return [d_W1, d_b1, d_W2, d_b2, d_W3, d_b3]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on_feedback(
        self,
        observations: np.ndarray,
        rewards: np.ndarray,
        epochs: int = 20,
    ) -> float:
        """
        Train on a batch of (observation, human_reward) pairs.

        Parameters
        ----------
        observations : (N, obs_dim)  states where the human gave feedback
        rewards      : (N,)          human reward values (+10 / -10)
        epochs       : number of gradient steps over the batch

        Returns
        -------
        Final MSE loss.
        """
        targets = rewards[:, None]   # (N, 1)
        loss = 0.0

        for _ in range(epochs):
            r_hat, cache = self._forward(observations)     # (N, 1)
            diff = r_hat - targets                         # (N, 1)
            loss = float(np.mean(diff ** 2))

            # dMSE/d(r_hat) = 2/N * diff
            d_r = (2.0 / len(observations)) * diff

            grads = self._backward(d_r, cache)

            # Adam update
            self._t += 1
            params = self._params()
            for i, (p, g) in enumerate(zip(params, grads)):
                self._m[i] = self._beta1 * self._m[i] + (1.0 - self._beta1) * g
                self._v[i] = self._beta2 * self._v[i] + (1.0 - self._beta2) * g ** 2
                m_hat = self._m[i] / (1.0 - self._beta1 ** self._t)
                v_hat = self._v[i] / (1.0 - self._beta2 ** self._t)
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self._eps)

        return loss

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, file_path: str | pathlib.Path) -> None:
        """Save weights and Adam state to a .npz file."""
        path = pathlib.Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
            lr=np.array(self.lr),
            adam_t=np.array(self._t),
            adam_m0=self._m[0], adam_m1=self._m[1],
            adam_m2=self._m[2], adam_m3=self._m[3],
            adam_m4=self._m[4], adam_m5=self._m[5],
            adam_v0=self._v[0], adam_v1=self._v[1],
            adam_v2=self._v[2], adam_v3=self._v[3],
            adam_v4=self._v[4], adam_v5=self._v[5],
        )
        print(f"HCRL reward model saved to {path}")

    @classmethod
    def load(cls, file_path: str | pathlib.Path) -> "HCRLRewardModel":
        """Load a saved HCRL reward model from a .npz file."""
        data = np.load(file_path)

        model = cls.__new__(cls)
        model.W1 = data["W1"]
        model.b1 = data["b1"]
        model.W2 = data["W2"]
        model.b2 = data["b2"]
        model.W3 = data["W3"]
        model.b3 = data["b3"]
        model.lr = float(data["lr"])

        model._beta1 = 0.9
        model._beta2 = 0.999
        model._eps   = 1e-8
        model._t     = int(data["adam_t"])
        model._m     = [data[f"adam_m{i}"] for i in range(6)]
        model._v     = [data[f"adam_v{i}"] for i in range(6)]

        print(f"HCRL reward model loaded from {file_path}")
        return model
