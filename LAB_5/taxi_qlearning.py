
import json
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
rng = np.random.default_rng(SEED)


EPISODES      = 5000     # training episodes
ALPHA         = 0.1      # learning rate
GAMMA         = 0.99     # discount factor
EPS_START     = 1.0      # initial exploration
EPS_MIN       = 0.05     # minimum exploration
EPS_DECAY     = 0.999    # per-episode multiplicative decay
MAX_STEPS     = 200      # step limit per episode (Taxi-v4 default)
EVAL_EPISODES = 1000     # greedy evaluation episodes

env = gym.make("Taxi-v4")
n_states  = env.observation_space.n   # 500
n_actions = env.action_space.n        # 6
Q = np.zeros((n_states, n_actions))


def epsilon_greedy(state, eps):
    if rng.random() < eps:
        return int(rng.integers(n_actions))
    # break ties randomly so the untrained table is not biased to action 0
    best = np.flatnonzero(Q[state] == Q[state].max())
    return int(rng.choice(best))


# ---------------- Training ----------------
rewards, steps, successes = [], [], []
eps = EPS_START
for ep in range(EPISODES):
    state, _ = env.reset(seed=SEED if ep == 0 else None)
    total, success = 0, False
    for t in range(MAX_STEPS):
        action = epsilon_greedy(state, eps)
        nxt, r, terminated, truncated, _ = env.step(action)
        # Q-Learning (off-policy) update: uses max over next actions
        target = r + (0.0 if terminated else GAMMA * Q[nxt].max())
        Q[state, action] += ALPHA * (target - Q[state, action])
        state = nxt
        total += r
        if terminated:
            success = True      # successful drop-off (+20)
        if terminated or truncated:
            break
    rewards.append(total)
    steps.append(t + 1)
    successes.append(success)
    eps = max(EPS_MIN, eps * EPS_DECAY)

# ---------------- Greedy evaluation ----------------
eval_rewards, eval_steps, eval_success = [], [], []
for _ in range(EVAL_EPISODES):
    state, _ = env.reset()
    total = 0
    for t in range(MAX_STEPS):
        action = int(np.argmax(Q[state]))
        state, r, terminated, truncated, _ = env.step(action)
        total += r
        if terminated or truncated:
            break
    eval_rewards.append(total)
    eval_steps.append(t + 1)
    eval_success.append(bool(terminated))

# ---------------- Metrics ----------------
rewards = np.array(rewards)
steps = np.array(steps)
successes = np.array(successes, dtype=float)


def block(a, lo, hi):
    return float(np.mean(a[lo:hi]))


results = {
    "episodes": EPISODES,
    "alpha": ALPHA, "gamma": GAMMA,
    "epsilon_start": EPS_START, "epsilon_min": EPS_MIN, "epsilon_decay": EPS_DECAY,
    "first_100_avg_reward": block(rewards, 0, 100),
    "first_100_avg_steps": block(steps, 0, 100),
    "first_100_success_rate": block(successes, 0, 100),
    "last_100_avg_reward": block(rewards, -100, None),
    "last_100_avg_steps": block(steps, -100, None),
    "last_100_success_rate": block(successes, -100, None),
    "training_avg_reward_all": float(rewards.mean()),
    "best_training_episode_reward": float(rewards.max()),
    "worst_training_episode_reward": float(rewards.min()),
    "eval_episodes": EVAL_EPISODES,
    "eval_avg_reward": float(np.mean(eval_rewards)),
    "eval_std_reward": float(np.std(eval_rewards)),
    "eval_avg_steps": float(np.mean(eval_steps)),
    "eval_success_rate": float(np.mean(eval_success)),
    "reward_by_500_episode_block": [block(rewards, i, i + 500) for i in range(0, EPISODES, 500)],
}
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)
np.save("q_table.npy", Q)
np.save("train_rewards.npy", rewards)

# ---------------- Plots ----------------
def moving_avg(x, w=100):
    return np.convolve(x, np.ones(w) / w, mode="valid")

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].plot(rewards, alpha=0.3, label="Episode reward")
ax[0].plot(np.arange(99, EPISODES), moving_avg(rewards), color="red", lw=2, label="100-episode moving avg")
ax[0].set_xlabel("Episode"); ax[0].set_ylabel("Total reward"); ax[0].set_title("Reward vs. Episodes")
ax[0].legend(); ax[0].grid(alpha=0.3)

ax[1].plot(np.arange(99, EPISODES), moving_avg(steps), color="green", lw=2)
ax[1].set_xlabel("Episode"); ax[1].set_ylabel("Steps per episode (100-ep avg)")
ax[1].set_title("Steps per Episode"); ax[1].grid(alpha=0.3)

ax[2].plot(np.arange(99, EPISODES), moving_avg(successes), color="purple", lw=2)
ax[2].set_xlabel("Episode"); ax[2].set_ylabel("Success rate (100-ep avg)")
ax[2].set_title("Task Completion Rate"); ax[2].set_ylim(0, 1.05); ax[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=150)

plt.figure(figsize=(8, 4.5))
plt.plot(rewards, alpha=0.3, label="Episode reward")
plt.plot(np.arange(99, EPISODES), moving_avg(rewards), color="red", lw=2, label="100-episode moving avg")
plt.xlabel("Episode"); plt.ylabel("Total reward")
plt.title("Q-Learning on Taxi-v4: Reward vs. Episodes")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("reward_vs_episodes.png", dpi=150)

# ---------------- Text rendering of one trained episode ----------------
env2 = gym.make("Taxi-v4", render_mode="ansi")
state, _ = env2.reset(seed=7)
frames = [env2.render()]
total = 0
for t in range(MAX_STEPS):
    state, r, terminated, truncated, _ = env2.step(int(np.argmax(Q[state])))
    total += r
    frames.append(env2.render())
    if terminated or truncated:
        break
with open("demo_episode.txt", "w") as f:
    for i, fr in enumerate(frames):
        f.write(f"--- step {i} ---\n{fr}\n")
    f.write(f"Total reward: {total}, steps: {t + 1}\n")

print(json.dumps(results, indent=2))
