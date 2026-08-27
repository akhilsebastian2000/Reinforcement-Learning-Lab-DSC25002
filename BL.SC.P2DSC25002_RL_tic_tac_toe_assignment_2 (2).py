import random
import numpy as np
from collections import defaultdict

class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [' '] * 9
        self.done = False
        self.winner = None
        return self.get_state()

    def get_state(self):
        return ''.join(self.board)

    def available_actions(self):
        return [i for i, x in enumerate(self.board) if x == ' ']

    def check_winner(self, player):
        wins = [(0,1,2),(3,4,5),(6,7,8),
                (0,3,6),(1,4,7),(2,5,8),
                (0,4,8),(2,4,6)]
        return any(all(self.board[i] == player for i in combo) for combo in wins)

    def step(self, action, player):
        self.board[action] = player

        if self.check_winner(player):
            self.done = True
            self.winner = player
            return self.get_state(), 1, True

        if ' ' not in self.board:
            self.done = True
            self.winner = None
            return self.get_state(), 0.5, True

        return self.get_state(), 0, False


class QLearningAgent:
    def __init__(self):
        self.q = defaultdict(lambda: np.zeros(9))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)

        q_values = self.q[state]
        valid_q = [(a, q_values[a]) for a in actions]
        max_q = max(valid_q, key=lambda x:x[1])[1]
        best = [a for a,q in valid_q if q==max_q]
        return random.choice(best)

    def update(self, state, action, reward, next_state, next_actions):
        future = 0
        if next_actions:
            future = max([self.q[next_state][a] for a in next_actions])

        self.q[state][action] += self.alpha * (
            reward + self.gamma * future - self.q[state][action]
        )


def random_move(env):
    return random.choice(env.available_actions())


def train(agent, episodes):

    for ep in range(episodes):

        env = TicTacToe()
        state = env.reset()

        while not env.done:

            actions = env.available_actions()
            action = agent.choose_action(state, actions)

            next_state, reward, done = env.step(action,'X')

            if done:
                agent.update(state,action,reward,next_state,[])
                break

            opponent = random_move(env)

            next_state2,reward2,done = env.step(opponent,'O')

            if done:

                if env.winner=='O':
                    reward=-1
                elif env.winner is None:
                    reward=0.5

                agent.update(state,action,reward,next_state2,[])
                break

            next_actions=env.available_actions()

            agent.update(state,action,0,next_state2,next_actions)

            state=next_state2


def evaluate(agent,games=1000):

    wins=0
    losses=0
    draws=0

    old_eps=agent.epsilon
    agent.epsilon=0

    for _ in range(games):

        env=TicTacToe()
        state=env.reset()

        while not env.done:

            action=agent.choose_action(state,env.available_actions())

            state,reward,done=env.step(action,'X')

            if done:
                break

            opponent=random_move(env)

            state,reward,done=env.step(opponent,'O')

        if env.winner=='X':
            wins+=1
        elif env.winner=='O':
            losses+=1
        else:
            draws+=1

    agent.epsilon=old_eps

    return wins/games*100,losses/games*100,draws/games*100


episodes=[100,500,1000,5000,10000]

for ep in episodes:

    agent=QLearningAgent()

    train(agent,ep)

    w,l,d=evaluate(agent)

    print(f"\nTraining Episodes: {ep}")
    print(f"Win : {w:.2f}%")
    print(f"Loss: {l:.2f}%")
    print(f"Draw: {d:.2f}%")