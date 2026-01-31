import os
import glob
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim

from simulation import Simulation
from BaseClassAnt import Ant
from randomant import RandomAnt

# -------------------------
# Config
# -------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "policy.pt")

ARENA_GLOB = os.path.join(os.path.dirname(__file__), "Training_Arenas", "*.txt")

# Training hyperparams
GAMMA = 0.99
LR = 5e-4 #3e-4
HIDDEN = 128

MAX_STEPS = 1400
GROUPS = 900   #800       # training groups
K = 4                 # episodes per group (before update)

ENTROPY_COEF = 0.01
CRITIC_COEF = 0.5

# league mix
P_RANDOM = 0.30
P_BASE   = 0.30
P_SNAP   = 0.40

SNAPSHOT_EVERY = 100 #50   # groups
MAX_SNAPSHOTS = 12

torch.set_num_threads(1)


# -------------------------
# Policy network (Actor-Critic)
# -------------------------
N_ACTIONS = 8  # 0..7
INPUT_DIM = 50


class PolicyNet(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN, n_actions=N_ACTIONS):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, n_actions)   # logits
        self.critic = nn.Linear(hidden_dim, 1)          # V(s)

    def forward(self, x):
        h = self.base(x)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        dist = torch.distributions.Categorical(logits=logits)
        return dist, value


def _clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


# -------------------------
# Shared observation encoder
# -------------------------
class ObsMixin:
    def _sense_int_grid(self, grid, default: float = 0.0):
        res = []
        W = self.simulation.gridWidth
        H = self.simulation.gridHeight
        for (dx, dy) in Ant.DIRECTIONS:
            nx = self.x + dx
            ny = self.y + dy
            if nx < 0 or ny < 0 or nx >= W or ny >= H:
                res.append(default)
            else:
                res.append(float(grid[nx][ny]))
        return res

    def _build_obs(self):
        has_food = 1.0 if self.hasFood else 0.0

        food = self._sense_int_grid(self.simulation.foodGrid, default=0.0)
        own_ants = self._sense_int_grid(self.simulation.antGrid[self.team], default=0.0)
        other_ants = self._sense_int_grid(self.simulation.antGrid[1 - self.team], default=0.0)
        obstacles = self._sense_int_grid(self.simulation.obstacleGrid, default=1.0)
        own_nest = self._sense_int_grid(self.simulation.nestGrid[self.team], default=0.0)
        other_nest = self._sense_int_grid(self.simulation.nestGrid[1 - self.team], default=0.0)

        own_pher = self._sense_int_grid(self.simulation.pheromoneGrid[self.team], default=0.0)
        other_pher = self._sense_int_grid(self.simulation.pheromoneGrid[1 - self.team], default=0.0)

        food = [_clip01(v / 9.0) for v in food]
        own_ants = [_clip01(v / 3.0) for v in own_ants]
        other_ants = [_clip01(v / 3.0) for v in other_ants]
        own_pher = [_clip01(v) for v in own_pher]
        other_pher = [_clip01(v) for v in other_pher]

        denom = max(1.0, float(self.simulation.maxId[self.team] - 1))
        id_norm = float(self.id) / denom

        last_onehot = [0.0] * N_ACTIONS
        la = int(getattr(self, "_last_action", Ant.A_IDLE))
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


# -------------------------
# Team reward (cached per tick, avoids double counting with 2-3 ants)
# -------------------------
def reward_signal_team(ant):
    sim = ant.simulation
    learner_team = sim.params["learner_team"]

    # init last food counts
    if not hasattr(sim, "_last_foodA"):
        sim._last_foodA = sim.computeFoodCount(Simulation.TEAM_A)
        sim._last_foodB = sim.computeFoodCount(Simulation.TEAM_B)

    # compute once per tick
    if (not hasattr(sim, "_cached_time")) or (sim._cached_time != sim.time):
        nowA = sim.computeFoodCount(Simulation.TEAM_A)
        nowB = sim.computeFoodCount(Simulation.TEAM_B)

        dA = nowA - sim._last_foodA
        dB = nowB - sim._last_foodB

        sim._last_foodA = nowA
        sim._last_foodB = nowB

        # zero-sum delta
        if learner_team == Simulation.TEAM_A:
            base = float(dA - dB)
        else:
            base = float(dB - dA)

        sim._cached_tick_reward = base
        sim._cached_time = sim.time

    # Opponent reward not needed for learning
    if ant.team != learner_team:
        return 0.0

    # Split team reward across learner ants to avoid multiplying by #ants
    n_learners = max(1, sim.maxId[learner_team])
    r = sim._cached_tick_reward / float(n_learners)

    # small step penalty
    r += (-0.01 / float(n_learners))

    # micro shaping (still not "behaviour", only reward)
    if not hasattr(ant, "_prev_hasFood"):
        ant._prev_hasFood = ant.hasFood

    own_nest_center = ant.senseOwnNest()[4] if hasattr(ant, "senseOwnNest") else 0
    just_picked = (ant.hasFood and (not ant._prev_hasFood))
    just_dropped = ((not ant.hasFood) and ant._prev_hasFood)

    if just_picked and own_nest_center == 0:
        r += (0.05 / float(n_learners))
    if just_dropped and own_nest_center == 1:
        r += (0.10 / float(n_learners))

    ant._prev_hasFood = ant.hasFood

    if sim.time >= sim.params.get("max_steps", MAX_STEPS):
        sim.terminate()

    return r


# -------------------------
# Ant classes for training
# -------------------------
class LearningAnt(Ant, ObsMixin):
    policy = None
    optimizer = None

    # shared buffers (team-level)
    log_probs = []
    values = []
    rewards = []
    entropies = []

    def __init__(self, simulation, x, y, team, id):
        super().__init__(simulation, x, y, team, id)
        self._last_action = Ant.A_IDLE

    def act(self):
        obs = self._build_obs()
        dist, value = self.policy(obs)

        # sample during training
        action = dist.sample()
        action_i = int(action.item())
        self._last_action = action_i

        # execute action semantics
        if action_i == Ant.A_PHEROMONE:
            if self.energy >= 1:
                self.energy -= 1
                self.dropPheromone()
            reward = self.simulation.rewardSignal(self)
        elif action_i == Ant.A_IDLE:
            if self.energy >= 1:
                self.energy -= 1
            reward = self.simulation.rewardSignal(self)
        else:
            reward = super().act(action_i)

        # small action costs to reduce spam / useless moves (training-only)
        if action_i == Ant.A_PHEROMONE:
            reward -= 0.02
        if action_i == Ant.A_IDLE:
            reward -= 0.005

        # store
        self.log_probs.append(dist.log_prob(action))
        self.entropies.append(dist.entropy())
        self.values.append(value)
        self.rewards.append(float(reward))

        return reward


class GreedyPolicyAnt(Ant, ObsMixin):
    policy = None

    def __init__(self, simulation, x, y, team, id):
        super().__init__(simulation, x, y, team, id)
        self._last_action = Ant.A_IDLE

    @torch.no_grad()
    def act(self):
        obs = self._build_obs()
        dist, _ = self.policy(obs)
        action_i = int(torch.argmax(dist.probs).item())
        self._last_action = action_i

        if action_i == Ant.A_PHEROMONE:
            if self.energy >= 1:
                self.energy -= 1
                self.dropPheromone()
            return self.simulation.rewardSignal(self)
        elif action_i == Ant.A_IDLE:
            if self.energy >= 1:
                self.energy -= 1
            return self.simulation.rewardSignal(self)
        else:
            return super().act(action_i)


# -------------------------
# Update step (A2C-style)
# -------------------------
def update_policy(policy: PolicyNet, optimizer: optim.Optimizer):
    if len(LearningAnt.rewards) == 0:
        return

    # discounted returns over the interleaved action sequence
    returns = []
    G = 0.0
    for r in reversed(LearningAnt.rewards):
        G = r + GAMMA * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)

    values = torch.stack(LearningAnt.values)
    log_probs = torch.stack(LearningAnt.log_probs)
    entropies = torch.stack(LearningAnt.entropies)

    advantages = returns - values

    actor_loss = -(log_probs * advantages.detach()).mean()
    critic_loss = advantages.pow(2).mean()
    entropy_loss = -ENTROPY_COEF * entropies.mean()

    loss = actor_loss + CRITIC_COEF * critic_loss + entropy_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # clear buffers
    LearningAnt.log_probs.clear()
    LearningAnt.values.clear()
    LearningAnt.rewards.clear()
    LearningAnt.entropies.clear()


# -------------------------
# League opponent management
# -------------------------
def available_arenas():
    paths = sorted(glob.glob(ARENA_GLOB))
    if not paths:
        raise FileNotFoundError(f"No arenas found with glob: {ARENA_GLOB}")
    return paths


def pick_arena(paths):
    return random.choice(paths)


def save_policy(policy: PolicyNet, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(policy.state_dict(), path)


def load_policy_from(path: str) -> PolicyNet:
    p = PolicyNet()
    p.load_state_dict(torch.load(path, map_location="cpu"))
    p.eval()
    return p


def run_episode(arena_path: str, learner_policy: PolicyNet, opponent_kind: str, opponent_policy: PolicyNet | None, learner_team: int):
    """
    One episode, learner on learner_team (0=A or 1=B).
    Opponent can be: 'random', 'base', 'snap' (policy).
    """
    # attach policies to classes
    LearningAnt.policy = learner_policy

    if opponent_kind in ("base", "snap"):
        GreedyPolicyAnt.policy = opponent_policy

    # choose antA / antB classes based on learner_team
    if learner_team == Simulation.TEAM_A:
        antA = LearningAnt
        antB = RandomAnt if opponent_kind == "random" else GreedyPolicyAnt
    else:
        antA = RandomAnt if opponent_kind == "random" else GreedyPolicyAnt
        antB = LearningAnt

    sim = Simulation(
        antA=antA,
        antB=antB,
        rewardSignal=reward_signal_team,
        params={"max_steps": MAX_STEPS, "learner_team": learner_team},
        arenaPath=arena_path,
        logfile=None,
    )

    while not sim.hasTerminated():
        sim.tick()
        if sim.time >= MAX_STEPS:
            break

    foodA = sim.computeFoodCount(Simulation.TEAM_A)
    foodB = sim.computeFoodCount(Simulation.TEAM_B)
    sim.shutdown()
    return foodA, foodB


# -------------------------
# Main training loop
# -------------------------
def main():
    random.seed(0)
    torch.manual_seed(0)

    arenas = available_arenas()

    learner = PolicyNet()
    optimizer = optim.Adam(learner.parameters(), lr=LR)

    # "base" opponent = snapshot of initial learner (starts weak, but good as fixed baseline)
    base_opponent = PolicyNet()
    base_opponent.load_state_dict(learner.state_dict())
    base_opponent.eval()

    snapshots = []  # list of (path, policy)

    print(f"Training on {len(arenas)} arenas (domain randomization via arena sampling).")
    print("Opponent mix: Random + Base + Snapshot League")

    for group in range(1, GROUPS + 1):
        sum_score = 0.0
        sum_foodA = 0.0
        sum_foodB = 0.0

        for _ in range(K):
            arena_path = pick_arena(arenas)

            # pick opponent
            p = random.random()
            if p < P_RANDOM:
                opp_kind = "random"
                opp_policy = None
            elif p < P_RANDOM + P_BASE:
                opp_kind = "base"
                opp_policy = base_opponent
            else:
                if snapshots:
                    opp_kind = "snap"
                    _, opp_policy = random.choice(snapshots)
                else:
                    opp_kind = "base"
                    opp_policy = base_opponent

            # run both sides to reduce side-bias
            for learner_team in (Simulation.TEAM_A, Simulation.TEAM_B):
                foodA, foodB = run_episode(arena_path, learner, opp_kind, opp_policy, learner_team)
                score = (foodA - foodB) if learner_team == Simulation.TEAM_A else (foodB - foodA)

                sum_score += score
                sum_foodA += foodA
                sum_foodB += foodB

        update_policy(learner, optimizer)

        avg_score = sum_score / float(2 * K)
        print(f"[Group {group:4d}] avg_score={avg_score:+.3f}  (foodA={sum_foodA/(2*K):.2f}, foodB={sum_foodB/(2*K):.2f})")

        # snapshots
        if group % SNAPSHOT_EVERY == 0:
            snap_path = os.path.join(MODEL_DIR, "snapshots", f"policy_g{group:04d}.pt")
            save_policy(learner, snap_path)
            snapshots.append((snap_path, load_policy_from(snap_path)))
            if len(snapshots) > MAX_SNAPSHOTS:
                snapshots.pop(0)
            print(f"  + snapshot saved: {snap_path}")

        # periodic save of final weights
        if group % 25 == 0:
            save_policy(learner, MODEL_PATH)

    save_policy(learner, MODEL_PATH)
    print(f"Done. Final policy saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
