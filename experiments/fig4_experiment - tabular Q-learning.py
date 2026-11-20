"""Figure 4: Independent tabular Q-learning on a periodic 10 x 10 lattice."""

import numpy as np
import random
from pathlib import Path

# Random seed configuration
SEED = 50


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


# Network and simulation parameters (matched to independent DQN)
rho = 1.0
R_const = 1.0
P_const = 0.0
SIZE = 10
N = SIZE * SIZE
total_round = 100000
focus_round = 5000

# Tabular Q-learning hyperparameters
ALPHA = 0.1
GAMMA = 0.99
OUTPUT_DIR = Path("results")


class IndependentTabularQLearning:
    def __init__(self, tau_init, tau_anneal_steps, alpha=ALPHA, gamma=GAMMA):
        # Each agent maintains its own zero-initialized 32-state, 2-action Q-table.
        self.q_tables = np.zeros((N, 32, 2), dtype=np.float64)
        self.agent_ids = np.arange(N)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.tau_init = float(tau_init)
        self.tau_final = 0.10
        self.tau_anneal_steps = int(tau_anneal_steps)
        self.tau_eval = 0.10
        self.tau = float(tau_init)
        self.env_steps = 0
        self.tau_log = []

    @staticmethod
    def encode_states(states):
        # Encode the five binary state features as an integer from 0 to 31.
        bits = states.astype(np.int64)
        return bits @ np.array([16, 8, 4, 2, 1], dtype=np.int64)

    def action_batch(self, states):
        state_ids = self.encode_states(states)
        q = self.q_tables[self.agent_ids, state_ids]  # (100, 2)
        logits = (q - q.max(axis=1, keepdims=True)) / max(self.tau, 1e-6)
        exp_logits = np.exp(logits)
        p = exp_logits[:, 0] / exp_logits.sum(axis=1)
        # Boltzmann action sampling: 0 = cooperate, 1 = defect.
        return (np.random.random(N) >= p).astype(np.int64)

    def update_batch(self, states, actions, rewards, next_states):
        s = self.encode_states(states)
        s_next = self.encode_states(next_states)
        q_old = self.q_tables[self.agent_ids, s, actions]
        next_max = self.q_tables[self.agent_ids, s_next].max(axis=1)
        target = rewards + self.gamma * next_max
        self.q_tables[self.agent_ids, s, actions] = q_old + self.alpha * (target - q_old)

    def on_env_step(self):
        self.env_steps += 1
        if self.env_steps <= self.tau_anneal_steps:
            ratio = self.env_steps / max(1, self.tau_anneal_steps)
            self.tau = self.tau_init + (self.tau_final - self.tau_init) * ratio
        else:
            self.tau = self.tau_final
        self.tau_log.append(self.tau)

    def set_eval_temperature(self):
        self.tau = self.tau_eval


class PDG_Vectorized:
    def __init__(self, rho, Dr: float, tau_init: float, anneal_step: int, alpha=ALPHA):
        self.rho = rho
        self.Dr = Dr
        self.cooperation_rates = []
        self.policy = IndependentTabularQLearning(
            tau_init=tau_init, tau_anneal_steps=anneal_step, alpha=alpha
        )

        R = R_const
        P = P_const
        S = -self.Dr
        T = 1.0 + self.Dr

        grid = np.arange(N, dtype=np.int32).reshape(SIZE, SIZE)
        up = np.roll(grid, +1, axis=0).ravel()
        right = np.roll(grid, -1, axis=1).ravel()
        down = np.roll(grid, -1, axis=0).ravel()
        left = np.roll(grid, +1, axis=1).ravel()
        self.neighbor = np.stack([up, right, down, left], axis=1)

        self.payoff = np.array([[R, S], [T, P]], dtype=np.float32)
        self.new_action = np.random.randint(0, 2, size=N, dtype=np.int8)
        self.last_state = np.zeros((N, 5), dtype=np.float32)
        self.new_state = np.zeros((N, 5), dtype=np.float32)
        self.rewards = np.zeros((N,), dtype=np.float32)
        self.compute_states()

    def compute_states(self):
        neighbor_new_actions = self.new_action[self.neighbor]
        self.new_state[:, :4] = neighbor_new_actions.astype(np.float32)
        self.new_state[:, 4] = self.new_action.astype(np.float32)

    def compute_rewards(self):
        me = self.new_action.astype(np.int64)
        neighbor_new_actions = self.new_action[self.neighbor].astype(np.int64)
        separate_reward = self.payoff[me[:, None], neighbor_new_actions]
        self.rewards = separate_reward.mean(axis=1).astype(np.float32)

    def compute_coop_rate(self):
        return (self.new_action == 0).mean()

    def run(self):
        self.cooperation_rates.append(self.compute_coop_rate())

        # Phase 1: 95,000 training steps with online one-step Q-learning updates.
        for step in range(total_round - focus_round):
            self.last_state = self.new_state.copy()
            self.new_action = self.policy.action_batch(self.last_state).astype(np.int8)
            self.compute_states()
            self.compute_rewards()
            self.policy.update_batch(
                self.last_state, self.new_action, self.rewards, self.new_state
            )
            self.policy.on_env_step()
            self.cooperation_rates.append(self.compute_coop_rate())

        # Phase 2: 5,000 evaluation steps with frozen Q-tables.
        cooper_cnt = 0.0
        self.policy.set_eval_temperature()
        for _ in range(focus_round):
            self.last_state = self.new_state.copy()
            self.new_action = self.policy.action_batch(self.last_state).astype(np.int8)
            self.compute_states()
            self.compute_rewards()
            current_coop_rate = self.compute_coop_rate()
            self.cooperation_rates.append(current_coop_rate)
            cooper_cnt += current_coop_rate

        K = int(self.policy.tau_anneal_steps // 2)
        if len(self.policy.tau_log) >= K:
            B_mean_tau = float(np.mean(self.policy.tau_log[:K]))
        else:
            B_mean_tau = float(np.mean(self.policy.tau_log))

        avg_focus = cooper_cnt / focus_round
        print(f"[Independent Tabular Q-learning] Dr: {self.Dr}, rho: {self.rho}, "
              f"SIZE: {N}, SEED: {SEED}, alpha: {self.policy.alpha}, "
              f"gamma: {self.policy.gamma}, tau_init: {self.policy.tau_init}, "
              f"tau_final: {self.policy.tau_final}, "
              f"tau_anneal_steps: {self.policy.tau_anneal_steps}, "
              f"B:{B_mean_tau:.6f}, result: {avg_focus:.6f}")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filename = OUTPUT_DIR / (
            f"tabular_independent_coop_SIZE{SIZE}_SEED{SEED}_Dr{self.Dr}"
            f"_alpha_{self.policy.alpha}_gamma_{self.policy.gamma}"
            f"_tauinit_{self.policy.tau_init}_taufinal_{self.policy.tau_final}"
            f"_anneal_{self.policy.tau_anneal_steps}_B_{B_mean_tau:.6f}.txt"
        )
        np.savetxt(filename, np.asarray(self.cooperation_rates), fmt="%.6f")


def sweep_Dr(dr_values=None, rho_val=1.0, seed_values=None,
             tau_init_values=None, anneal_steps=None, alpha=ALPHA):
    if dr_values is None:
        dr_values = [0.25]
    if seed_values is None:
        seed_values = list(range(195, 210))
    if tau_init_values is None:
        tau_init_values = [0.127, 0.260, 0.393, 0.527, 0.660, 0.793]
    if anneal_steps is None:
        anneal_steps = [95000]

    for seed in seed_values:
        for tau_init in tau_init_values:
            for dr in dr_values:
                for anneal in anneal_steps:
                    global SEED
                    SEED = seed
                    set_global_seed(SEED)
                    pdg = PDG_Vectorized(
                        rho=rho_val, Dr=dr, tau_init=tau_init,
                        anneal_step=anneal, alpha=alpha
                    )
                    pdg.run()


if __name__ == "__main__":
    sweep_Dr()