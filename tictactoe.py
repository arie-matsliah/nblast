import random
from collections import defaultdict

# Tic-Tac-Toe Environment
class TicTacToe:
    def __init__(self):
        self.reset()

    def reset(self):
        self.board = [" "] * 9
        self.current_player = "X"
        return self.get_state()

    def get_state(self):
        return "".join(self.board)

    def available_actions(self):
        return [i for i, v in enumerate(self.board) if v == " "]

    def step(self, action):
        # Place mark
        self.board[action] = self.current_player
        winner = self.check_winner()
        done = winner is not None or " " not in self.board
        reward = 0

        if done:
            if winner == self.current_player:  # the one who just moved
                reward = 1
            elif winner is not None:  # the other player won
                reward = -1
            else:  # draw
                reward = 0

        # Switch player
        self.current_player = "O" if self.current_player == "X" else "X"
        return self.get_state(), reward, done

    def check_winner(self):
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
            (0, 4, 8), (2, 4, 6)              # diagonals
        ]
        for a, b, c in lines:
            if self.board[a] == self.board[b] == self.board[c] != " ":
                return self.board[a]
        return None

    def render(self):
        print("\n".join([
            "|".join(self.board[i:i+3]) for i in range(0, 9, 3)
        ]))
        print()

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, epsilon=0.2, alpha=0.5, gamma=0.9):
        self.q = defaultdict(float)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_q(self, state, action):
        return self.q[(state, action)]

    def best_action(self, state, actions):
        qs = [self.get_q(state, a) for a in actions]
        max_q = max(qs)
        best = [a for a, q in zip(actions, qs) if q == max_q]
        return random.choice(best)

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        return self.best_action(state, actions)

    def update(self, state, action, reward, next_state, next_actions, done):
        old_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            next_q = max([self.get_q(next_state, a) for a in next_actions]) if next_actions else 0
            target = reward + self.gamma * next_q
        self.q[(state, action)] = old_q + self.alpha * (target - old_q)

# Training loop
def train(episodes):
    env = TicTacToe()
    agent = QLearningAgent()
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            actions = env.available_actions()
            action = agent.choose_action(state, actions)

            # Save which player is moving
            current_player = env.current_player

            next_state, reward, done = env.step(action)
            next_actions = env.available_actions()

            # Update ONLY for the player who just moved
            agent.update(state, action, reward, next_state, next_actions, done)

            state = next_state
    return agent

# Play against the trained agent
def play(agent):
    env = TicTacToe()
    state = env.reset()
    env.render()

    while True:
        if env.current_player == "X":
            # Agent move
            action = agent.best_action(state, env.available_actions())
            state, reward, done = env.step(action)
            print("Agent's move:")
            env.render()
            if done:
                if reward == 1:
                    print("Agent wins!")
                elif reward == -1:
                    print("You win!")
                else:
                    print("Draw!")
                break
        else:
            # Human move
            action = int(input("Your move (0-8): "))
            while action not in env.available_actions():
                action = int(input("Invalid. Your move (0-8): "))
            state, reward, done = env.step(action)
            env.render()
            if done:
                if reward == 1:
                    print("Agent wins!")
                elif reward == -1:
                    print("You win!")
                else:
                    print("Draw!")
                break

if __name__ == "__main__":
    print("Training agent... please wait")
    trained_agent = train(episodes=50000)
    print("Training finished. Let's play!")
    play(trained_agent)
