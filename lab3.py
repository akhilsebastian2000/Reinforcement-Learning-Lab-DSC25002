import numpy as np
import random

# ─────────────────────────────
# GridWorld Setup (5x5 grid)
# ─────────────────────────────
GRID_SIZE = 5
NUM_STATES = GRID_SIZE * GRID_SIZE  # 25 states
ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
ACTION_MAP = {'UP': (-1,0), 'DOWN': (1,0), 'LEFT': (0,-1), 'RIGHT': (0,1)}
GOAL_STATE = (4, 4)
START_STATE = (0, 0)
GAMMA = 0.9      # discount factor
THETA = 0.0001   # convergence threshold

def state_to_idx(r, c): return r * GRID_SIZE + c
def idx_to_state(idx): return (idx // GRID_SIZE, idx % GRID_SIZE)

def step(state, action):
    r, c = state
    dr, dc = ACTION_MAP[action]
    nr, nc = r + dr, c + dc
    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
        next_state = (nr, nc)
    else:
        next_state = state  # stay if out of bounds
    reward = 0 if next_state == GOAL_STATE else -1
    done = next_state == GOAL_STATE
    return next_state, reward, done

# ─────────────────────────────
# Task 1: Environment Summary
# ─────────────────────────────
print("=" * 55)
print("TASK 1: GridWorld Environment Formulation")
print("=" * 55)
print(f"States       : {NUM_STATES} states (5x5 grid, (0,0) to (4,4))")
print(f"Actions      : {ACTIONS}")
print(f"Reward       : -1 per step, 0 at goal")
print(f"Terminal     : Goal state (4,4)")
print(f"Agent Goal   : Reach (4,4) with max reward\n")

# ─────────────────────────────
# Task 2: Policy Evaluation
# ─────────────────────────────
print("=" * 55)
print("TASK 2: Policy Evaluation (Random Policy)")
print("=" * 55)

# Random policy: equal probability for all actions
def random_policy(state):
    return {a: 0.25 for a in ACTIONS}

V = np.zeros(NUM_STATES)

for iteration in range(1, 6):
    delta = 0
    V_new = V.copy()
    for idx in range(NUM_STATES):
        state = idx_to_state(idx)
        if state == GOAL_STATE:
            continue
        v = 0
        for action, prob in random_policy(state).items():
            next_state, reward, _ = step(state, action)
            next_idx = state_to_idx(*next_state)
            v += prob * (reward + GAMMA * V[next_idx])
        V_new[idx] = v
        delta = max(delta, abs(V_new[idx] - V[idx]))
    V = V_new
    status = "Converged" if delta < THETA else "Not Converged"
    print(f"Iteration {iteration}: Max Delta = {delta:.6f}  [{status}]")

print("\nValue Function after 5 iterations (random policy):")
print(V.reshape(GRID_SIZE, GRID_SIZE).round(3))

# Full convergence
for _ in range(1000):
    delta = 0
    V_new = V.copy()
    for idx in range(NUM_STATES):
        state = idx_to_state(idx)
        if state == GOAL_STATE:
            continue
        v = sum(prob * (step(state, a)[1] + GAMMA * V[state_to_idx(*step(state, a)[0])])
                for a, prob in random_policy(state).items())
        V_new[idx] = v
        delta = max(delta, abs(V_new[idx] - V[idx]))
    V = V_new
    if delta < THETA:
        break

# ─────────────────────────────
# Task 3: Value Iteration
# ─────────────────────────────
print("\n" + "=" * 55)
print("TASK 3: Value Iteration & Optimal Policy")
print("=" * 55)

V_opt = np.zeros(NUM_STATES)

for _ in range(10000):
    delta = 0
    V_new = V_opt.copy()
    for idx in range(NUM_STATES):
        state = idx_to_state(idx)
        if state == GOAL_STATE:
            continue
        values = []
        for action in ACTIONS:
            next_state, reward, _ = step(state, action)
            values.append(reward + GAMMA * V_opt[state_to_idx(*next_state)])
        V_new[idx] = max(values)
        delta = max(delta, abs(V_new[idx] - V_opt[idx]))
    V_opt = V_new
    if delta < THETA:
        break

# Derive optimal policy
def optimal_action(state):
    best_val, best_act = float('-inf'), None
    for action in ACTIONS:
        next_state, reward, _ = step(state, action)
        val = reward + GAMMA * V_opt[state_to_idx(*next_state)]
        if val > best_val:
            best_val, best_act = val, action
    return best_act

print("\nOptimal Value Function:")
print(V_opt.reshape(GRID_SIZE, GRID_SIZE).round(3))

print("\nOptimal Policy Grid (arrows):")
arrow = {'UP':'↑','DOWN':'↓','LEFT':'←','RIGHT':'→'}
for r in range(GRID_SIZE):
    row = ""
    for c in range(GRID_SIZE):
        if (r,c) == GOAL_STATE:
            row += " G "
        else:
            row += f" {arrow[optimal_action((r,c))]} "
    print(row)

# Print table for S1-S5
print("\nState | Optimal Action | State Value")
print("-" * 40)
sample_states = [(0,0),(0,4),(2,2),(4,0),(3,3)]
for i, s in enumerate(sample_states, 1):
    print(f"S{i} {s} | {optimal_action(s):>6} | {V_opt[state_to_idx(*s)]:.4f}")

# ─────────────────────────────
# Task 4: Path Comparison
# ─────────────────────────────
print("\n" + "=" * 55)
print("TASK 4: Policy Comparison")
print("=" * 55)

def run_policy(policy_fn, max_steps=100):
    state = START_STATE
    path, total_reward, steps = [state], 0, 0
    for _ in range(max_steps):
        action = policy_fn(state)
        next_state, reward, done = step(state, action)
        total_reward += reward
        steps += 1
        path.append(next_state)
        state = next_state
        if done:
            return path, steps, total_reward, True
    return path, steps, total_reward, False

# Random policy runner
def random_policy_fn(state): return random.choice(ACTIONS)

# Evaluated policy (greedy from V)
def evaluated_policy_fn(state):
    best_val, best_act = float('-inf'), None
    for action in ACTIONS:
        next_state, reward, _ = step(state, action)
        val = reward + GAMMA * V[state_to_idx(*next_state)]
        if val > best_val:
            best_val, best_act = val, action
    return best_act

random.seed(42)
r_path, r_steps, r_reward, r_done = run_policy(random_policy_fn)
e_path, e_steps, e_reward, e_done = run_policy(evaluated_policy_fn)
o_path, o_steps, o_reward, o_done = run_policy(optimal_action)

print(f"\n{'Policy':<20} {'Steps':>6} {'Total Reward':>14} {'Goal Reached':>13}")
print("-" * 55)
print(f"{'Random Policy':<20} {r_steps:>6} {r_reward:>14} {str(r_done):>13}")
print(f"{'Evaluated Policy':<20} {e_steps:>6} {e_reward:>14} {str(e_done):>13}")
print(f"{'Optimal Policy':<20} {o_steps:>6} {o_reward:>14} {str(o_done):>13}")

print(f"\nOptimal path: {o_path}")