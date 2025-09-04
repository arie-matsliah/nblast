import random
from collections import defaultdict
import time  # To see training time


# (The TicTacToe and QLearningAgent classes are unchanged from the previous version)
# ...

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
        self.board[action] = self.current_player
        winner = self.check_winner()
        done = winner is not None or " " not in self.board
        reward = 0
        if done and winner == self.current_player:
            reward = 1
        self.current_player = "O" if self.current_player == "X" else "X"
        return self.get_state(), reward, done

    def check_winner(self):
        lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for a, b, c in lines:
            if self.board[a] == self.board[b] == self.board[c] != " ":
                return self.board[a]
        return None

    def render(self):
        board_rows = [" | ".join(self.board[i:i + 3]) for i in range(0, 9, 3)]
        print("\n" + "\n---|---|---\n".join(board_rows) + "\n")


class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.q = defaultdict(float)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def get_q(self, state, action):
        return self.q.get((state, action), 0.0)

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
            target = reward - self.gamma * next_q
        self.q[(state, action)] = old_q + self.alpha * (target - old_q)


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
            agent.update(state, action, reward, next_state, next_actions, done)
            state = next_state
    return agent


# --- NEW CODE STARTS HERE ---

## Optimal Minimax Agent
class MinimaxAgent:
    """
    A perfect Tic-Tac-Toe agent that uses Minimax with memoization (caching)
    to avoid re-calculating moves for the same board state.
    """
    def __init__(self, player):
        self.player = player
        self.cache = {}  # NEW: Initialize the cache

    def choose_action(self, board, available_actions):
        # NEW: Check the cache before computing
        # We convert the board list to a tuple so it can be used as a dictionary key.
        board_tuple = tuple(board)
        if board_tuple in self.cache:
            return self.cache[board_tuple]

        # If not in cache, calculate the best move
        if self.player == 'X':
            best_score = -float('inf')
            best_move = None
            for action in available_actions:
                board[action] = 'X'
                score = self.minimax(board, 0, False)
                board[action] = ' '
                if score > best_score:
                    best_score = score
                    best_move = action
        else: # 'O'
            best_score = float('inf')
            best_move = None
            for action in available_actions:
                board[action] = 'O'
                score = self.minimax(board, 0, True)
                board[action] = ' '
                if score < best_score:
                    best_score = score
                    best_move = action

        # NEW: Store the result in the cache before returning
        self.cache[board_tuple] = best_move
        return best_move

    def minimax(self, board, depth, is_maximizing):
        # (The core minimax logic is unchanged)
        winner = self.check_winner(board)
        if winner == 'X': return 10 - depth
        if winner == 'O': return depth - 10
        if ' ' not in board: return 0

        if is_maximizing:
            best_score = -float('inf')
            for i in range(9):
                if board[i] == ' ':
                    board[i] = 'X'
                    score = self.minimax(board, depth + 1, False)
                    board[i] = ' '
                    best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for i in range(9):
                if board[i] == ' ':
                    board[i] = 'O'
                    score = self.minimax(board, depth + 1, True)
                    board[i] = ' '
                    best_score = min(score, best_score)
            return best_score

    def check_winner(self, board):
        lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]
        for a, b, c in lines:
            if board[a] == board[b] == board[c] != ' ':
                return board[a]
        return None


## Test Function
def run_test(trained_agent, optimal_agent, num_games):
    """
    Plays a number of games between the trained agent ('X') and the optimal agent ('O').
    """
    stats = {"Trained Agent (X) Wins": 0, "Optimal Agent (O) Wins": 0, "Draw": 0}
    env = TicTacToe()

    for i in range(num_games):
        state = env.reset()
        done = False
        while not done:
            if env.current_player == "X":
                actions = env.available_actions()
                action = trained_agent.best_action(state, actions)
                state, _, done = env.step(action)
            else:  # Player "O"
                # The MinimaxAgent needs the raw board and actions, not the string state
                actions = env.available_actions()
                action = optimal_agent.choose_action(env.board, actions)
                state, _, done = env.step(action)

        # Tally results
        winner = env.check_winner()
        if winner == "X":
            stats["Trained Agent (X) Wins"] += 1
        elif winner == "O":
            stats["Optimal Agent (O) Wins"] += 1
        else:
            stats["Draw"] += 1

    for key, value in stats.items():
        if value:
            print(f"{key}: {value} / {num_games} ({(value / num_games) * 100:.1f}%)")

    return stats["Draw"] / num_games

def ascii_bar_plot(values, width=50):
    """
    Draws a simple horizontal bar plot using ASCII.
    values: list of floats between 0 and 1
    width: maximum bar width in characters
    """
    for v in values:
        # Scale value to width
        bar_length = int(v * width)
        bar = "#" * bar_length
        print(f"{v:>4.2f} | {bar}")

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
    interactive = True
    if interactive:
        while True:
            print("Training agent... this may take a moment.")
            trained_agent = train(episodes=20000)
            print("Training finished. Let's play!")
            play(trained_agent)
    else:
        results = []
        for epochs in range(0, 30000, 1500):
            start_time = time.time()
            print(f"\nTraining Q-Learning agent with {epochs} episodes...")
            trained_agent = train(episodes=epochs)
            print(f"Training finished in {time.time() - start_time:.2f} seconds.")
            optimal_agent = MinimaxAgent(player='O')
            res = run_test(trained_agent, optimal_agent, num_games=1000)
            results.append(res)
        ascii_bar_plot(results)
