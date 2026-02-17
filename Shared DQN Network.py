import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import os

# === 隨機數設定 ===
SEED = 50

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def randf():
    return random.random()

def randi(lim):
    return random.randrange(lim)

rho = 1.0

R_const = 1.0
P_const = 0.0

SIZE = 30
N = SIZE * SIZE
total_round = 100000
focus_round = 5000
train_steps_per_env_step = 1 

class QNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  
        return self.fc2(x)         

class SharedDQN:
    def __init__(
        self,
        N_env = N,
        state_size=5,
        action_size=2,
        lr=1e-4,
        gamma=0.99,
        memory_size=90_000,
        batch_size=256,
        target_update_frequency=2000,
        device=None,
        hidden_size=96,
        warmup_steps = 0,

        n_step=5,

        tau_init=1.5,
        tau_final=0.10,
        tau_anneal_steps=95000,
        tau_eval=0.10, 
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.warmup_steps = int(warmup_steps)

        self.model = QNet(state_size, action_size, hidden_size).to(self.device)
        self.target_model = QNet(state_size, action_size, hidden_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.AdamW(self.model.parameters(), lr, weight_decay=1e-4)
        self.criterion = nn.SmoothL1Loss(reduction="none") 

        self.memory_capacity = memory_size
        self.memory_size = 0
        self.memory_ptr = 0

        self.memory_state     = np.zeros((memory_size, state_size), dtype=np.float32)
        self.memory_action    = np.full((memory_size,), -1, dtype=np.int64)
        self.memory_reward    = np.zeros((memory_size,), dtype=np.float32)
        self.memory_new_state = np.zeros((memory_size, state_size), dtype=np.float32)

        self.memory_agent = np.full((memory_size,), -1, dtype=np.int32)
        self.memory_step  = np.full((memory_size,), -1, dtype=np.int32)
        self.N_env = N_env

        self.batch_size = batch_size
        self.env_steps = 0
        self.target_update_frequency = target_update_frequency

        # --- Softmax ---
        self.tau_init = float(tau_init)
        self.tau_final = float(tau_final)
        self.tau_anneal_steps = int(tau_anneal_steps)
        self.tau_eval = float(tau_eval)
        self.tau = float(tau_init)

        self.tau_log = [] 

        self.n_step = n_step

    def argmax_random_tie_break(self, logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        max_per_row = logits.max(dim=1, keepdim=True).values

        is_candidate = logits >= (max_per_row - eps)

        random_scores = torch.rand_like(logits)
        random_scores = torch.where(is_candidate, random_scores, torch.full_like(logits, -1.0))

        chosen_indices = random_scores.argmax(dim=1, keepdim=True)

        return chosen_indices

    @torch.no_grad()
    def action_batch(self, states_np: np.ndarray) -> np.ndarray:
        states_t = torch.from_numpy(states_np).float().to(self.device, non_blocking=True)
        q = self.model(states_t)  

        q_centered = q - q.max(dim=1, keepdim=True).values
        logits = q_centered / max(self.tau, 1e-6)
        probabilities = torch.softmax(logits, dim=1)  

        actions = torch.multinomial(probabilities, num_samples=1).squeeze(1).cpu().numpy()
        return actions.astype(np.int64)
    
    def _anneal_tau(self):
        if self.env_steps <= self.tau_anneal_steps:
            ratio = self.env_steps / max(1, self.tau_anneal_steps)
            self.tau = self.tau_init + (self.tau_final - self.tau_init) * ratio
        else:
            self.tau = self.tau_final

    def remember_batch(self, states, actions, rewards, next_states):
        B = states.shape[0]

        index = (np.arange(B) + self.memory_ptr) % self.memory_capacity

        self.memory_state[index]     = states
        self.memory_action[index]    = actions
        self.memory_reward[index]    = rewards
        self.memory_new_state[index] = next_states

        self.memory_agent[index] = np.arange(B, dtype=np.int32)
        self.memory_step[index]  = self.env_steps

        self.memory_ptr  = (self.memory_ptr + B) % self.memory_capacity
        self.memory_size = int(min(self.memory_size + B, self.memory_capacity))

    def replay(self):
        if self.memory_size < max(self.batch_size, self.warmup_steps):
            return

        capacity = self.memory_capacity
        B = self.batch_size
        n = self.n_step
        
        if self.memory_size < (n * (self.N_env or 0) + 1):
            n = 1

        index = np.random.randint(0, self.memory_size, size=B)

        offsets = (np.arange(n, dtype=np.int64) * (self.N_env if self.N_env is not None else 0))[None, :] 
        index_sequence = (index[:, None] + offsets) % capacity  

        steps_sequence  = self.memory_step[index_sequence]  
        agents_sequence = self.memory_agent[index_sequence] 

        written_mask = (self.memory_step[index_sequence] != -1)

        valid = written_mask & (agents_sequence == agents_sequence[:, :1]) & (steps_sequence == (steps_sequence[:, :1] + np.arange(n)[None, :]))
        full_valid = valid.all(axis=1)

        if full_valid.any():
            isNstep = np.where(full_valid)[0]
            index_sequence_isNstep = index_sequence[isNstep] 

            state  = torch.from_numpy(self.memory_state[index_sequence_isNstep[:, 0]]).to(self.device)
            action  = torch.from_numpy(self.memory_action[index_sequence_isNstep[:, 0]]).to(self.device).unsqueeze(1)
            new_state = torch.from_numpy(self.memory_new_state[index_sequence_isNstep[:, -1]]).to(self.device)

            rewards = torch.from_numpy(self.memory_reward[index_sequence_isNstep]).to(self.device) 
            gammas = torch.pow(torch.full((n,), self.gamma, device=self.device), torch.arange(n, device=self.device)).view(1, n)
            Discount_Reward = (rewards * gammas).sum(dim=1, keepdim=True)  

            q_current = self.model(state).gather(1, action)
            with torch.no_grad():
                q_online_n = self.model(new_state)                    
                next_max_action = self.argmax_random_tie_break(q_online_n)
                q_next = self.target_model(new_state).gather(1, next_max_action)            
                target = Discount_Reward + (self.gamma ** n) * q_next

            loss_per = self.criterion(q_current, target)
            loss_n = loss_per.mean()
        else:
            loss_n = None

        is1step = np.where(~full_valid)[0]
        if is1step.size > 0:
            index_1step = index[is1step]
            state_1step  = torch.from_numpy(self.memory_state[index_1step]).to(self.device)
            action_1step  = torch.from_numpy(self.memory_action[index_1step]).to(self.device).unsqueeze(1)
            reward_1step  = torch.from_numpy(self.memory_reward[index_1step]).to(self.device).unsqueeze(1)
            new_state_1step = torch.from_numpy(self.memory_new_state[index_1step]).to(self.device)

            q_current_1step = self.model(state_1step).gather(1, action_1step)
            with torch.no_grad():
                q_online_1 = self.model(new_state_1step)                          
                next_max_action_1step = self.argmax_random_tie_break(q_online_1)       
                q_next_1step = self.target_model(new_state_1step).gather(1, next_max_action_1step)
                target_1step = reward_1step + self.gamma * q_next_1step

            loss_per1 = self.criterion(q_current_1step, target_1step)
            loss_1 = loss_per1.mean()
        else:
            loss_1 = None

        Nstep_cnt = int(full_valid.sum())            
        onestep_cnt = int((~full_valid).sum())     
        total = Nstep_cnt + onestep_cnt

        loss = None
        if (loss_n is not None) and (loss_1 is not None):
            loss = (Nstep_cnt * loss_n + onestep_cnt * loss_1) / max(1, total)
        elif loss_n is not None:
            loss = loss_n
        elif loss_1 is not None:
            loss = loss_1
        else:
            return  

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()

    def on_env_step(self):
        self.env_steps += 1

        if (self.env_steps > self.warmup_steps):
            self._anneal_tau()

        self.tau_log.append(self.tau)

        if self.target_update_frequency > 0 and (self.env_steps % self.target_update_frequency == 0):
            self.target_model.load_state_dict(self.model.state_dict())


class PDG_Vectorized:
    def __init__(self, rho, Dr: float, tau_init: float, anneal_step:int):
        self.rho = rho
        self.Dr = Dr
        self.cooperation_rates = []
        self.policy = SharedDQN(tau_init=tau_init, tau_anneal_steps=anneal_step)

        R = R_const
        P = P_const
        S = -self.Dr
        T = 1.0 + self.Dr

        grid = np.arange(N, dtype=np.int32).reshape(SIZE, SIZE)
        up    = np.roll(grid,  +1, axis=0).ravel() 
        right = np.roll(grid,  -1, axis=1).ravel()  
        down  = np.roll(grid,  -1, axis=0).ravel()  
        left  = np.roll(grid,  +1, axis=1).ravel() 
        self.neighbor = np.stack([up, right, down, left], axis=1) 

        self.payoff = np.array([[R, S],
                                [T, P]], dtype=np.float32)

        self.new_action  = np.random.randint(0, 2, size=N, dtype=np.int8)                

        self.last_state = np.zeros((N, 5), dtype=np.float32) 
        self.new_state  = np.zeros((N, 5), dtype=np.float32)  
        self.rewards    = np.zeros((N,),  dtype=np.float32)  

        self.compute_states()   
        
    def compute_states(self):
        neighbor_new_actions = self.new_action[self.neighbor]
        self.new_state[:, :4] = neighbor_new_actions.astype(np.float32)   
        self.new_state[:,  4] = self.new_action.astype(np.float32)

    def compute_rewards(self):
        me = self.new_action.astype(np.int64)           
        neighbor_new_actions = self.new_action[self.neighbor].astype(np.int64)  

        seperate_reward = self.payoff[me[:, None], neighbor_new_actions]               
        reward = seperate_reward.mean(axis=1).astype(np.float32) 
        self.rewards = reward

    def compute_coop_rate(self):
        return (self.new_action == 0).mean()

    def run(self):
        self.cooperation_rates.append(self.compute_coop_rate())

        for step in range(total_round - focus_round):
            self.last_state = self.new_state.copy() 
            self.new_action = self.policy.action_batch(self.last_state).astype(np.int8)
            self.compute_states()     
            self.compute_rewards() 

            self.policy.remember_batch(self.last_state, self.new_action, self.rewards , self.new_state)

            if step >= self.policy.warmup_steps and (step % 4 == 0):
                self.policy.replay()

            self.policy.on_env_step()


            coop_rate = self.compute_coop_rate()
            self.cooperation_rates.append(coop_rate)

        cooper_cnt = 0.0
        self.policy.tau = self.policy.tau_eval
        
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
        print(f"Dr: {self.Dr}, rho: {self.rho}, SIZE: {N}, SEED: {SEED}, "
              f"hidden_layer: 1, tau_init: {self.policy.tau_init}, tau_final: {self.policy.tau_final}, "
              f"tau_anneal_steps: {self.policy.tau_anneal_steps}, B:{B_mean_tau:.6f}, result: {avg_focus:.6f}")

        os.makedirs("results", exist_ok=True)
        filename = (
            f"results/coop_SIZE{SIZE}_SEED{SEED}_hidden_layer1_Dr{self.Dr}"
            f"_tauinit_{self.policy.tau_init}_taufinal_{self.policy.tau_final}"
            f"_anneal_{self.policy.tau_anneal_steps}_B_{B_mean_tau:.6f}.txt"
        )
        np.savetxt(filename, np.array(self.cooperation_rates), fmt="%.6f")

def sweep_Dr(
    dr_values=None,
    rho_val=1.0,
    seed_values=None, 
    tau_init_values=None,  
    anneal_steps = None
):
    if dr_values is None:
        dr_values = np.round(np.arange(0.10, 0.40 + 1e-9, 0.01), 3)
    if seed_values is None:
        seed_values = list(range(195, 225))
    if tau_init_values is None:
        tau_init_values = [0.127, 0.193, 0.260, 0.327, 0.393, 0.460, 0.527, 0.593, 0.660, 0.727]
    if anneal_steps is None:
        anneal_steps = [95000]

    for seed in seed_values:
        for tau_init in tau_init_values:
            for dr in dr_values:
                for anneal in anneal_steps:
                    global SEED
                    SEED = seed
                    set_global_seed(SEED)
                    pdg = PDG_Vectorized(rho=rho_val, Dr=dr, tau_init=tau_init, anneal_step = anneal)
                    pdg.run()
sweep_Dr()