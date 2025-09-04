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

        # The reward is for the player who just moved
        reward = 0
        if done:
            if winner == self.current_player:
                reward = 1  # Win
            # No reward for a loss (-1) is needed here, as the learning
            # algorithm will penalize the losing move implicitly.
            # A draw results in the default reward of 0.

        # Switch player for the next turn
        self.current_player = "O" if self.current_player == "X" else "X"
        return self.get_state(), reward, done

    def check_winner(self):
        lines = [
            (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
            (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
            (0, 4, 8), (2, 4, 6)  # diagonals
        ]
        for a, b, c in lines:
            if self.board[a] == self.board[b] == self.board[c] != " ":
                return self.board[a]
        return None

    def render(self):
        board_rows = [" | ".join(self.board[i:i + 3]) for i in range(0, 9, 3)]
        print("\n" + "\n---|---|---\n".join(board_rows) + "\n")


# Q-Learning Agent
class QLearningAgent:
    def __init__(self, epsilon=0.2, alpha=0.5, gamma=0.9):
        self.q = defaultdict(float)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

    def best_action(self, state, actions):
        qs = [self.get_q(state, a) for a in actions]
        max_q = max(qs)
        # In case of a tie, pick one randomly
        best = [a for a, q in zip(actions, qs) if q == max_q]
        return random.choice(best)

    def choose_action(self, state, actions):
        if random.random() < self.epsilon:
            return random.choice(actions)
        return self.best_action(state, actions)

    def update(self, state, action, reward, next_state, next_actions, done):
        """
        FIX #1: The Q-learning update rule was corrected for an adversarial game.
        """
        old_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            # The value of the next state is the NEGATIVE of the max Q-value
            # for the opponent. The opponent will choose the move that is best
            # for them, which is worst for the current agent.
            next_q = max([self.get_q(next_state, a) for a in next_actions]) if next_actions else 0
            target = reward - self.gamma * next_q  # The crucial change is this minus sign

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

            next_state, reward, done = env.step(action)
            next_actions = env.available_actions()

            # Update Q-table for the move that was just made
            agent.update(state, action, reward, next_state, next_actions, done)

            state = next_state
    return agent


# Play against the trained agent
def play(agent):
    """
    FIX #2: The play loop was rewritten to be more robust and to
    correctly announce the winner.
    """
    env = TicTacToe()
    state = env.reset()
    done = False
    print("Starting a new game. You are 'O'.")
    env.render()

    while not done:
        actions = env.available_actions()
        if not actions:
            break  # Game ends in a draw if no actions left

        if env.current_player == "X":
            # Agent's turn
            print("Agent's move (X):")
            action = agent.best_action(state, actions)
            state, _, done = env.step(action)
        else:
            # Human's turn
            action = -1
            while action not in actions:
                try:
                    action = int(input(f"Your move (O) - choose from {actions}: "))
                    if action not in actions:
                        print("Invalid move. Please choose from the available spots.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            state, _, done = env.step(action)

        env.render()

    # Announce the result once the game is over
    winner = env.check_winner()
    if winner:
        print(f"Game over! Player {winner} wins!")
    else:
        print("Game over! It's a draw!")


if __name__ == "__main__":
    while True:
        print("Training agent... this may take a moment.")
        trained_agent = train(episodes=50000)
        print("Training finished. Let's play!")
        play(trained_agent)