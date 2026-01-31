import os
import torch
import torch.nn as nn

from BaseClassAnt import Ant


# -------------------------
# Config
# -------------------------
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "policy.pt")
_CACHED_POLICY = None

N_ACTIONS = 8  # 0..7  (LEFT,RIGHT,UP,DOWN,TAKE,DROP,IDLE,PHEROMONE)

# -------------------------
# Policy Network (inference)
# -------------------------


class PolicyNet(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        h = self.base(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value


def _load_policy_once():
    global _CACHED_POLICY
    if _CACHED_POLICY is not None:
        return _CACHED_POLICY

    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {_MODEL_PATH}")

    state = torch.load(_MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError("Loaded model is not a state_dict dict.")

    # infer dims
    if "base.0.weight" not in state:
        raise RuntimeError("Checkpoint missing base.0.weight (cannot infer input/hidden dims).")
    hidden_dim = state["base.0.weight"].shape[0]
    input_dim = state["base.0.weight"].shape[1]

    # infer num_actions from checkpoint
    if "actor.weight" not in state:
        raise RuntimeError("Checkpoint missing actor.weight (cannot infer num_actions).")
    num_actions = state["actor.weight"].shape[0]

    policy = PolicyNet(input_dim=input_dim, hidden_dim=hidden_dim, num_actions=num_actions)
    policy.load_state_dict(state, strict=True)
    policy.eval()

    _CACHED_POLICY = policy
    return _CACHED_POLICY


def _clip01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


class Ant44651225(Ant):
    """
    Final ant:
    - Just inference
    - NO hard-coded behaviour
    """

    def __init__(self, simulation, x, y, team, id):
        super().__init__(simulation, x, y, team, id)
        self._policy = _load_policy_once()
        self._last_action = Ant.A_IDLE

    # ---- Safe sensing (bounds-safe, avoids negative-index wrap) ----
    def _sense_int_grid(self, grid, default: float = 0.0):
        res = []
        W = self.simulation.gridWidth
        H = self.simulation.gridHeight
        for (dx, dy) in Ant.DIRECTIONS:  # L,R,U,D,C
            nx = self.x + dx
            ny = self.y + dy
            if nx < 0 or ny < 0 or nx >= W or ny >= H:
                res.append(default)
            else:
                res.append(float(grid[nx][ny]))
        return res

    def _sense_float_grid(self, grid, default: float = 0.0):
        # same as above but for float grids (pheromones)
        return self._sense_int_grid(grid, default=default)

    def _build_obs(self) -> torch.Tensor:
        # Base sensors
        has_food = 1.0 if self.hasFood else 0.0

        food = self._sense_int_grid(self.simulation.foodGrid, default=0.0)                 # 5
        own_ants = self._sense_int_grid(self.simulation.antGrid[self.team], default=0.0)  # 5
        other_ants = self._sense_int_grid(self.simulation.antGrid[1 - self.team], default=0.0)  # 5
        obstacles = self._sense_int_grid(self.simulation.obstacleGrid, default=1.0)       # 5 (OOB treated as obstacle)
        own_nest = self._sense_int_grid(self.simulation.nestGrid[self.team], default=0.0) # 5
        other_nest = self._sense_int_grid(self.simulation.nestGrid[1 - self.team], default=0.0) # 5

        own_pher = self._sense_float_grid(self.simulation.pheromoneGrid[self.team], default=0.0)       # 5
        other_pher = self._sense_float_grid(self.simulation.pheromoneGrid[1 - self.team], default=0.0) # 5

        # Normalizations
        # food: digits typically 0..9 -> scale by 9
        food = [_clip01(v / 9.0) for v in food]
        # ants: you said max ~3 -> scale by 3
        own_ants = [_clip01(v / 3.0) for v in own_ants]
        other_ants = [_clip01(v / 3.0) for v in other_ants]
        # pheromones already 0..1
        own_pher = [_clip01(v) for v in own_pher]
        other_pher = [_clip01(v) for v in other_pher]

        # id feature
        denom = max(1.0, float(self.simulation.maxId[self.team] - 1))
        id_norm = float(self.id) / denom

        # last action one-hot
        last_onehot = [0.0] * N_ACTIONS
        la = int(self._last_action)
        if 0 <= la < N_ACTIONS:
            last_onehot[la] = 1.0

        obs = (
            [has_food]
            + food
            + own_ants
            + other_ants
            + obstacles
            + own_nest
            + other_nest
            + own_pher
            + other_pher
            + [id_norm]
            + last_onehot
        )

        return torch.tensor(obs, dtype=torch.float32)

    @torch.no_grad()
    def _select_action(self, obs: torch.Tensor) -> int:
        expected = int(self._policy.base[0].in_features)
        assert obs.numel() == expected, f"Obs dim {obs.numel()} != expected {expected}"

        dist, _ = self._policy(obs)

        action = int(dist.sample().item())

        return max(0, min(N_ACTIONS - 1, action))

    def act(self):
        obs = self._build_obs()
        action = self._select_action(obs)
        self._last_action = action

        # Execute chosen action
        if action == Ant.A_PHEROMONE:
            # make pheromone cost an action/energy
            if self.energy >= 1:
                self.energy -= 1
                self.dropPheromone()
            return self.simulation.rewardSignal(self)

        if action == Ant.A_IDLE:
            if self.energy >= 1:
                self.energy -= 1
            return self.simulation.rewardSignal(self)

        # 0..5 handled by base class
        return super().act(action)