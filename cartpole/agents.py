"""
CartPole agent implementations.

Agents
------
RandomActionAgent  — random baseline
QLearningAgent     — tabular Q-Learning (used by baseline, HCRL, and RLHF)
VITAMERAgent       — non-myopic TAMER with Q_H value function (VI-TAMER)

State discretisation shared between QLearningAgent and VITAMERAgent is
factored into _DiscretizationMixin to avoid code duplication.
"""

import abc
import pathlib

import numpy as np

from cartpole import config as cfg
from cartpole.entities import Action, Observation, Reward, State

RandomStateType = np.random.RandomState | np.random.Generator


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Agent(abc.ABC):
    @abc.abstractmethod
    def begin_episode(self, observation: Observation) -> Action:
        pass

    @abc.abstractmethod
    def act(self, observation: Observation, reward: Reward) -> Action:
        pass


# ---------------------------------------------------------------------------
# Shared discretisation (QLearningAgent and VITAMERAgent both need this)
# ---------------------------------------------------------------------------

class _DiscretizationMixin:
    """
    Mixin that converts a 4-dimensional continuous CartPole observation into
    a single integer state index suitable for tabular Q-Learning.

    Uses uniform binning with NUM_BINS bins per feature, covering the
    physically meaningful range of each CartPole state variable.
    """

    _state_bins: list[np.ndarray]
    _max_bins:   int
    _num_states: int

    def _setup_discretization(self, num_bins: int = cfg.NUM_BINS) -> None:
        self._state_bins = [
            self._bin_range(-2.4, 2.4, num_bins),   # cart position
            self._bin_range(-3.0, 3.0, num_bins),   # cart velocity
            self._bin_range(-0.5, 0.5, num_bins),   # pole angle
            self._bin_range(-2.0, 2.0, num_bins),   # pole angular velocity
        ]
        self._max_bins  = max(len(b) for b in self._state_bins)
        self._num_states = (self._max_bins + 1) ** len(self._state_bins)

    @staticmethod
    def _bin_range(lo: float, hi: float, n: int) -> np.ndarray:
        return np.linspace(lo, hi, n + 1)[1:-1]

    def _obs_to_state(self, observation: Observation) -> State:
        return sum(
            int(np.digitize(feature, self._state_bins[i])) * ((self._max_bins + 1) ** i)
            for i, feature in enumerate(observation)
        )

    def _save_bins(self, path: pathlib.Path, q_array: np.ndarray, key: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            **{key: q_array},
            state_bins_0=self._state_bins[0],
            state_bins_1=self._state_bins[1],
            state_bins_2=self._state_bins[2],
            state_bins_3=self._state_bins[3],
        )

    def _load_bins(self, data: np.lib.npyio.NpzFile) -> None:
        self._state_bins = [data[f"state_bins_{i}"] for i in range(4)]
        self._max_bins   = max(len(b) for b in self._state_bins)


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

class RandomActionAgent(Agent):
    """Agent that always selects a random action — useful as a lower bound."""

    def __init__(self, random_state: RandomStateType | None = None):
        self.random_state: RandomStateType = random_state or np.random.default_rng()

    def begin_episode(self, observation: Observation) -> Action:
        return int(self.random_state.choice([0, 1]))

    def act(self, observation: Observation, reward: Reward) -> Action:
        return int(self.random_state.choice([0, 1]))


# ---------------------------------------------------------------------------
# Tabular Q-Learning  (used by baseline, HCRL, RLHF)
# ---------------------------------------------------------------------------

class QLearningAgent(_DiscretizationMixin, Agent):
    """
    Tabular Q-Learning agent with ε-greedy exploration and linear state discretisation.

    Shared across all training paradigms (baseline, HCRL, RLHF) so that only
    the reward signal changes between methods, not the learning algorithm.

    References
    ----------
    Sutton & Barto (2018), Chapter 6 — Temporal-Difference Learning
    """

    def __init__(
        self,
        learning_rate:          float = cfg.AGENT_LR,
        discount_factor:        float = cfg.AGENT_DISCOUNT,
        exploration_rate:       float = cfg.AGENT_EXPLORE,
        exploration_decay_rate: float = cfg.AGENT_DECAY,
        random_state:           RandomStateType | None = None,
    ):
        self.learning_rate          = learning_rate
        self.discount_factor        = discount_factor
        self.exploration_rate       = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.random_state: RandomStateType = random_state or np.random.default_rng()

        self.state:  State  | None = None
        self.action: Action | None = None

        self._setup_discretization()
        self._num_actions = 2
        self._q = np.zeros((self._num_states, self._num_actions))

    def begin_episode(self, observation: Observation) -> Action:
        self.exploration_rate *= self.exploration_decay_rate
        self.state  = self._obs_to_state(observation)
        self.action = int(np.argmax(self._q[self.state]))
        return self.action

    def act(self, observation: Observation, reward: Reward) -> Action:
        next_state = self._obs_to_state(observation)

        enable_exploration = (1 - self.exploration_rate) <= self.random_state.uniform(0, 1)
        next_action = (
            int(self.random_state.integers(0, self._num_actions))
            if enable_exploration
            else int(np.argmax(self._q[next_state]))
        )

        # Q-Learning update
        self._q[self.state, self.action] += self.learning_rate * (
            reward
            + self.discount_factor * np.max(self._q[next_state])
            - self._q[self.state, self.action]
        )

        self.state  = next_state
        self.action = next_action
        return next_action

    def save(self, file_path: str | pathlib.Path) -> None:
        path = pathlib.Path(file_path)
        self._save_bins(path, self._q, "q_table")
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, file_path: str | pathlib.Path) -> "QLearningAgent":
        data  = np.load(file_path)
        agent = cls.__new__(cls)
        agent._load_bins(data)
        agent._q          = data["q_table"]
        agent._num_states = agent._q.shape[0]
        agent._num_actions = agent._q.shape[1]
        agent.exploration_rate = 0.0  # greedy during evaluation
        agent.random_state     = np.random.default_rng()
        # Provide defaults so the object is fully functional
        agent.learning_rate          = cfg.AGENT_LR
        agent.discount_factor        = cfg.AGENT_DISCOUNT
        agent.exploration_decay_rate = cfg.AGENT_DECAY
        agent.state  = None
        agent.action = None
        print(f"Model loaded from {file_path}")
        return agent


# ---------------------------------------------------------------------------
# VI-TAMER  [HCRL §III-A-2 — Knox & Stone 2012]
# ---------------------------------------------------------------------------

class VITAMERAgent(_DiscretizationMixin, Agent):
    """
    VI-TAMER: Non-myopic TAMER with a tabular value function Q_H(s,a).

    Reference: Knox & Stone (2012), "Reinforcement Learning from Human Reward
    and Advice"
    Survey: Li et al. (2019), Section III-A-2

    Difference from plain TAMER (γ = 0):
      TAMER    → policy = argmax_a R̂_H(s,a)          [immediate only]
      VI-TAMER → Q_H(s,a) ← Q_H(s,a) + α·(R̂_H(s,a) + γ·max Q_H(s') - Q_H(s,a))
                 policy    = argmax_a Q_H(s,a)          [discounted future value]

    Setting γ = 0 exactly recovers plain TAMER because the future-value term
    disappears and Q_H(s,a) ≡ R̂_H(s,a).
    """

    def __init__(
        self,
        learning_rate:          float = cfg.AGENT_LR,
        discount_factor:        float = cfg.VI_TAMER_DISCOUNT,
        exploration_rate:       float = cfg.AGENT_EXPLORE,
        exploration_decay_rate: float = cfg.AGENT_DECAY,
        random_state:           RandomStateType | None = None,
    ) -> None:
        self.learning_rate          = learning_rate
        self.discount_factor        = discount_factor
        self.exploration_rate       = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.random_state: RandomStateType = random_state or np.random.default_rng()

        self.state:  State  | None = None
        self.action: Action | None = None

        self._setup_discretization()
        self._num_actions = 2
        # Q_H(s,a): value table driven by the human reward model R̂_H
        self._q_h = np.zeros((self._num_states, self._num_actions))

    def begin_episode(self, observation: Observation) -> Action:
        self.exploration_rate *= self.exploration_decay_rate
        self.state  = self._obs_to_state(observation)
        self.action = int(np.argmax(self._q_h[self.state]))
        return self.action

    def act(self, observation: Observation, reward: Reward) -> Action:
        """
        Standard act() — treats `reward` as the full TD signal directly.
        Prefer act_vi() when integrating R̂_H and future Q_H separately.
        """
        next_state = self._obs_to_state(observation)

        enable_exploration = (1 - self.exploration_rate) <= self.random_state.uniform(0, 1)
        next_action = (
            int(self.random_state.integers(0, self._num_actions))
            if enable_exploration
            else int(np.argmax(self._q_h[next_state]))
        )

        # VI-TAMER TD update (eq. 10 in survey)
        td_target = reward + self.discount_factor * np.max(self._q_h[next_state])
        self._q_h[self.state, self.action] += self.learning_rate * (
            td_target - self._q_h[self.state, self.action]
        )

        self.state  = next_state
        self.action = next_action
        return next_action

    def act_vi(
        self,
        obs: Observation,
        next_obs: Observation,
        reward_model,
        env_reward: float,
    ) -> Action:
        """
        VI-TAMER update with explicit R̂_H + γ·Q_H decomposition.

        The immediate signal comes from R̂_H(s_t, a_t); the future value from
        γ·max_a' Q_H(s_{t+1}, a'), matching eqs. (9-11) of the survey.

        Parameters
        ----------
        obs         : current observation (s_t)
        next_obs    : next observation (s_{t+1})
        reward_model: HCRLRewardModel (or None before first training)
        env_reward  : environment reward used as fallback when model is None
        """
        next_state = self._obs_to_state(next_obs)

        r_h = float(reward_model.predict(np.array(obs))) if reward_model is not None else env_reward

        td_target = r_h + self.discount_factor * np.max(self._q_h[next_state])
        self._q_h[self.state, self.action] += self.learning_rate * (
            td_target - self._q_h[self.state, self.action]
        )

        enable_exploration = (1 - self.exploration_rate) <= self.random_state.uniform(0, 1)
        next_action = (
            int(self.random_state.integers(0, self._num_actions))
            if enable_exploration
            else int(np.argmax(self._q_h[next_state]))
        )

        self.state  = next_state
        self.action = next_action
        return next_action

    def save(self, file_path: str | pathlib.Path) -> None:
        path = pathlib.Path(file_path)
        self._save_bins(path, self._q_h, "q_h_table")
        print(f"VI-TAMER model saved to {path}")

    @classmethod
    def load(cls, file_path: str | pathlib.Path) -> "VITAMERAgent":
        data  = np.load(file_path)
        agent = cls.__new__(cls)
        agent._load_bins(data)
        agent._q_h         = data["q_h_table"]
        agent._num_states  = agent._q_h.shape[0]
        agent._num_actions = agent._q_h.shape[1]
        agent.exploration_rate       = 0.0
        agent.learning_rate          = cfg.AGENT_LR
        agent.discount_factor        = cfg.VI_TAMER_DISCOUNT
        agent.exploration_decay_rate = cfg.AGENT_DECAY
        agent.random_state           = np.random.default_rng()
        agent.state  = None
        agent.action = None
        print(f"VI-TAMER model loaded from {file_path}")
        return agent
